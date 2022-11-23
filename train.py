# original author: signatrix
# adapted from https://github.com/signatrix/efficientdet/blob/master/train.py
# modified by Zylo117

import argparse
import datetime
import os
import traceback
from time import gmtime, strftime
import pytz
import cv2
import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.autonotebook import tqdm
from matplotlib import pyplot as plt

from backbone import EfficientDetBackbone
from efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater, FakeCocoDataset
from efficientdet.loss import FocalLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import postprocess, invert_affine, display



class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def get_args():
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    parser.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int, default=12, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=8, help='The number of images per batch among all devices')
    parser.add_argument('--head_only', type=boolean_string, default=False,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=75)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--data_path', type=str, default='datasets/', help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--debug', type=boolean_string, default=False,
                        help='whether visualize the predicted boxes of training, '
                             'the output images will be in test/')
    parser.add_argument('--call_cancer_only', type=boolean_string, default=True,
                        help='only train and test cancer')
    parser.add_argument('--use_paste_aug', type=boolean_string, default=True,
                        help='use copy and paste instance augmentation')
    parser.add_argument('--train_annFile', type=str, default=None)
    parser.add_argument('--train_image_dir', type=str, default=None)
    parser.add_argument('--val_annFile', type=str, default=None)
    parser.add_argument('--val_image_dir', type=str, default=None)
    parser.add_argument('--gpu_id', type=str, default=None)
    parser.add_argument('-ft', '--finetune', type=boolean_string, default=False,
                        help='fine-tune')
    args = parser.parse_args()
    return args


class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        # print('regression', regression.size())
        # print('classification', classification.size())
        # print('anchors', anchors.size())
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss

        
def get_current_time():
    tz = pytz.timezone('US/Eastern')
    current_time = datetime.datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
    return str(current_time)

def train(opt):
    params = Params(f'projects/{opt.project}.yml')

    

    if params.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if not opt.gpu_id is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
        params.num_gpus = len(opt.gpu_id.split(','))
    print(f'using {params.num_gpus}, ID is {opt.gpu_id}')

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        cuda = True
    else:
        cuda = False
        torch.manual_seed(42)

    current_time = get_current_time()
    opt.log_path = opt.log_path + f'/{params.project_name}/{current_time}'
    tensorboard_dir = opt.log_path + f'/tensorboard'
    weight_save_dir = opt.log_path + f'/weight'
    temp_cal_dir = opt.log_path + f'/temp'
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(weight_save_dir, exist_ok=True)
    os.makedirs(temp_cal_dir, exist_ok=True)
    
    print('current_time' + ':' + current_time + '\n')

    logfile = os.path.join(opt.log_path, 'parameters.txt')

    p=vars(opt)
    log_file = open(logfile, "a+")
    log_file.write('current_time' + ':' + current_time + '\n')
    log_file.write('\n')
    for key, val in p.items():
        log_file.write(key + ':' + str(val) + '\n')
    log_file.write('\n')
    log_file.write('\n')
    # current_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

    p=vars(params)
    for key, val in p.items():
        log_file.write(key + ':' + str(val) + '\n')
    log_file.write('\n')

    log_file.close()# 

    opt.train_annFile = params.train_annFile
    opt.train_image_dir = params.train_image_dir
    opt.val_annFile = params.val_annFile
    opt.val_image_dir = params.val_image_dir

    training_params = {'batch_size': opt.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': opt.num_workers}

    val_params = {'batch_size': opt.batch_size,
                  'shuffle': True,
                  'drop_last': False,
                  'collate_fn': collater,
                  'num_workers': opt.num_workers}

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

    # training_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.train_set,
    #                            transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
    #                                                          Augmenter(),
    #                                                          Resizer(input_sizes[opt.compound_coef])]))

    # val_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.val_set,
    #                       transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
    #                                                     Resizer(input_sizes[opt.compound_coef])]))

    training_set = FakeCocoDataset(transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                             Augmenter(),
                                                             Resizer(input_sizes[opt.compound_coef])]),
                                                             args=opt, img_dir=opt.train_image_dir, annFile= opt.train_annFile, split = 'train')
    training_generator = DataLoader(training_set, **training_params)
    
    if os.path.exists(opt.val_annFile) and os.path.exists(opt.val_image_dir):
        val_set = FakeCocoDataset(transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                            Resizer(input_sizes[opt.compound_coef])]), 
                                                            args=opt,  img_dir=opt.val_image_dir, annFile= opt.val_annFile, split = 'val')

        val_generator = DataLoader(val_set, **val_params)
    else:
        val_set = None
        val_generator = None


    model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=opt.compound_coef,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales), load_weights=False)
    
    if params.num_gpus > 0:
        model = model.cuda()
        if params.num_gpus > 1:
            model = CustomDataParallel(model, params.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)

    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    writer = SummaryWriter(tensorboard_dir)

    
    
    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    # model = ModelWithLoss(model, debug=opt.debug)
    criterion = FocalLoss()

    # if cuda:
    #     model.cuda()
    # load last weights
    if opt.resume is not None and os.path.exists(opt.resume):
        checkpoint = torch.load(opt.resume, map_location=torch.device('cpu'))
        last_epoch = checkpoint['epoch']

        if isinstance(model, CustomDataParallel):
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])


        print("=> loaded checkpoint '{}' (epoch {})"
                .format(opt.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if opt.finetune:
            last_epoch = -1
            best_pred = 0.0
            best_loss = float('inf')
            print('fine-tune.................')
        else:
            last_epoch = float(checkpoint['epoch'])
            best_pred = float(checkpoint['best_pred'])
            best_loss = float(checkpoint['best_loss'])
            optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        last_epoch = -1
        best_pred = 0.0
        best_loss = float('inf')
        print('[Info] initializing weights...')
        
        init_weights(model)
    

    # freeze backbone if train head_only
    if opt.head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if params.num_gpus > 1 and opt.batch_size // params.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    
    model.train()
    num_iter_per_epoch_tr = len(training_generator)
    num_iter_per_epoch_val = len(val_generator)
    print(f'training num_iter_per_epoch: {num_iter_per_epoch_tr}')
    try:
        for epoch in range(opt.num_epochs):
            # last_epoch = step // num_iter_per_epoch
            if epoch <= last_epoch:
                continue

            epoch_loss = []
            epoch_loss_regression= []
            epoch_loss_classification = []
            tr_progress_bar = tqdm(training_generator)
            for iter, data in enumerate(tr_progress_bar):
                # if iter < step - last_epoch * num_iter_per_epoch_tr:
                #     progress_bar.update()
                #     continue
                if opt.debug and iter > 3:
                    break
                step = iter + num_iter_per_epoch_tr * epoch
                try:
                    imgs = data['img']
                    annot = data['annot']
                    # print('imgs size', imgs.size(), 'annot size', annot.size(), 'annot', annot[0])
                    # imgs size torch.Size([8, 3, 768, 768]) annot size torch.Size([8, 2, 5]) 
                    # annot tensor([[331.2207, 393.9860, 597.3701, 599.9611,   0.0000], [106.6851, 185.5393, 183.5513, 351.8619,   0.0000]])
                    # imgs size torch.Size([8, 3, 768, 768]) annot size torch.Size([8, 2, 5]) 
                    # annot tensor([[-1., -1., -1., -1., -1.],[-1., -1., -1., -1., -1.]])
                    if params.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    optimizer.zero_grad()
                    _, regression, classification, anchors = model(imgs)
                    cls_loss, reg_loss = criterion(classification, regression, anchors, annot)
                    # cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
                    
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()
                    loss = cls_loss + reg_loss
                    # loss = loss.mean()
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                    epoch_loss.append(float(loss))
                    # print('Train/Loss.....................', loss, step)
                    tr_progress_bar.set_description(
                        'Train Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.3f}. Reg loss: {:.3f}. Total loss: {:.3f}'.format(
                            epoch, opt.num_epochs, iter + 1, num_iter_per_epoch_tr, cls_loss.item(),
                            reg_loss.item(), loss.item()))
                    writer.add_scalar('Train/Loss', loss.item(), step)
                    
                    writer.add_scalar('Train/Loss_reg', reg_loss.item(), step)
                    writer.add_scalar('Train/Loss_cls', cls_loss.item(), step)


                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)

                    epoch_loss_classification.append(cls_loss.item())
                    epoch_loss_regression.append(reg_loss.item())

                    # if step % opt.save_interval == 0 and step > 0:
                    #     save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth', weight_save_dir)
                    #     print('checkpoint...')
                
                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue
            ep_cls_loss = np.mean(epoch_loss_classification)
            ep_reg_loss = np.mean(epoch_loss_regression)
            ep_loss = cls_loss + reg_loss
            writer.add_scalar('Train/Epoch_loss', ep_loss, epoch)
            writer.add_scalar('Train/Epoch_loss_reg', ep_reg_loss, epoch)
            writer.add_scalar('Train/Epoch_loss_cls', ep_cls_loss, epoch)

            figure = plot_img_pre_gt(imgs, regression, classification, anchors, params.obj_list, annots = annot)
            writer.add_figure(f"Train/{epoch}/{iter}",figure)
            scheduler.step(np.mean(epoch_loss))

            save_checkpoints(model, {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'best_pred': best_pred,
                'best_loss': best_loss,
                }, filename = f'efficientdet-d{opt.compound_coef}.pth.tar', saved_path= weight_save_dir)

            
            if not val_set is None:
                step = iter + num_iter_per_epoch_val * epoch
                if epoch % opt.val_interval == 0:
                    model.eval()
                    loss_regression_ls = []
                    loss_classification_ls = []
                    val_progress_bar = tqdm(val_generator)

                    for iter, data in enumerate(val_progress_bar):
                        if opt.debug and iter > 3:
                            break
                        with torch.no_grad():
                            imgs = data['img']
                            annot = data['annot']

                            if params.num_gpus == 1:
                                imgs = imgs.cuda()
                                annot = annot.cuda()
                                
                            _, regression, classification, anchors = model(imgs)
                            cls_loss, reg_loss = criterion(classification, regression, anchors, annot)
                            # cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
                            loss = cls_loss + reg_loss
                            cls_loss = cls_loss.mean()
                            reg_loss = reg_loss.mean()

                            
                            if loss == 0 or not torch.isfinite(loss):
                                continue
                            
                            val_progress_bar.set_description(
                                'Val Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.3f}. Reg loss: {:.3f}. Total loss: {:.3f}'.format(
                                    epoch, opt.num_epochs, iter + 1, num_iter_per_epoch_tr, cls_loss.item(),
                                    reg_loss.item(), loss.item()))
                            writer.add_scalar('Val/Loss', loss.item(), step)
                            writer.add_scalar('Val/Loss_reg', reg_loss.item(), step)
                            writer.add_scalar('Val/Loss_cls', cls_loss.item(), step)
                            
                            loss_classification_ls.append(cls_loss.item())
                            loss_regression_ls.append(reg_loss.item())
                    figure = plot_img_pre_gt(imgs, regression, classification, anchors, params.obj_list, annots = annot)
                    writer.add_figure(f"Val/{epoch}/{iter}",figure)

                    ep_cls_loss = np.mean(loss_classification_ls)
                    ep_reg_loss = np.mean(loss_regression_ls)
                    ep_loss = cls_loss + reg_loss

                    writer.add_scalar('Val/Epoch_loss', ep_loss, epoch)
                    writer.add_scalar('Val/Epoch_loss_reg', ep_reg_loss, epoch)
                    writer.add_scalar('Val/Epoch_loss_cls', ep_cls_loss, epoch)
                    
                    if ep_loss + opt.es_min_delta < best_loss:
                        best_loss = ep_loss
                        best_epoch = epoch

                        # save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth', weight_save_dir)
                        save_checkpoints(model, {
                        'epoch': epoch,
                        'optimizer': optimizer.state_dict(),
                        'best_pred': best_pred,
                        'best_loss': best_loss,
                        }, filename = f'efficientdet-d{opt.compound_coef}_best.pth.tar', saved_path= weight_save_dir)

                    model.train()

                    # Early stopping
                    if epoch - best_epoch > opt.es_patience > 0:
                        print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
                        break
    except KeyboardInterrupt:
        save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth', weight_save_dir)
        writer.close()
    writer.close()


def save_checkpoint(model, name, saved_path):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.state_dict(), os.path.join(saved_path, name))
    else:
        torch.save(model.state_dict(), os.path.join(saved_path, name))

def save_checkpoints(model, state, filename='checkpoint.pth.tar', saved_path = None):
    if isinstance(model, CustomDataParallel):
        state['state_dict'] = model.module.state_dict()
        torch.save(state, os.path.join(saved_path, filename))
    else:
        state['state_dict'] = model.state_dict()
        torch.save(state, os.path.join(saved_path, filename))

    filename = os.path.join(saved_path, filename)
    torch.save(state, filename)
    # if is_best:   
    #     best_model_filepath = os.path.join(saved_path, 'model_best.pth.tar')
    #     torch.save(state, best_model_filepath)

# debug
def plot_img_pre_gt(imgs, regressions, classifications, anchors, obj_list, annots = None, training = True, save_dir = None):
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    out = postprocess(imgs.detach(),
                        torch.stack([anchors[0]] * imgs.shape[0], 0).detach(), regressions.detach(), classifications.detach(),
                        regressBoxes, clipBoxes,
                        0.5, 0.3)
    imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
    imgs = ((imgs * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
    # imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
    imgs_with_bboxes = display(out, imgs, obj_list, imshow=False, training=training, annots = annots, save_dir = save_dir)
    num = len(imgs_with_bboxes)
    if not imgs_with_bboxes is None:
        figure = plt.figure(figsize=(6*num, 6))
        for i in range(num):
            plt.subplot(1, num, i+1)
            plt.imshow(imgs_with_bboxes[i])
        return figure
    else:
        return None


if __name__ == '__main__':
    opt = get_args()
    train(opt)
