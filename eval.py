# Author: Zylo117

"""
COCO-Style Evaluations

put images here datasets/your_project_name/val_set_name/*.jpg
put annotations here datasets/your_project_name/annotations/instances_{val_set_name}.json
put weights here /path/to/your/weights/*.pth
change compound_coef

"""

import json
import os

import argparse
import torch
import yaml
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string
from utils.utils import get_current_time

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='erosiveulcer_fine', help='project file that contains parameters')
ap.add_argument('-c', '--compound_coef', type=int, default=2, help='coefficients of efficientdet')
ap.add_argument('-w', '--weights', type=str, default=None, help='/path/to/weights')
ap.add_argument('--nms_threshold', type=float, default=0.5, help='nms threshold, don\'t change it if not for testing purposes')
ap.add_argument('--cuda', type=boolean_string, default=True)
ap.add_argument('--device', type=int, default=2)
ap.add_argument('--float16', type=boolean_string, default=False)
ap.add_argument('--override', type=boolean_string, default=True, help='override previous bbox results file if exists')
args = ap.parse_args()

compound_coef = args.compound_coef
nms_threshold = args.nms_threshold
use_cuda = args.cuda
gpu = args.device
use_float16 = args.float16
override_prev_results = args.override
project_name = args.project

print(f'running coco-style evaluation on project {project_name}')

params = yaml.safe_load(open(f'projects/{project_name}.yml'))
obj_list = params['obj_list']
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

log_dict = dict()
log_dict['obj_list'] = obj_list

def infer_forward(model, img_dir, threshold=0.05, set_name = None):
    results = []

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    img_names = os.listdir(img_dir)

    for image_name in tqdm(img_names):
        img_filepath = os.path.join(img_dir, image_name)
        ori_imgs, framed_imgs, framed_metas = preprocess(img_filepath, max_size=input_sizes[compound_coef], mean=params['mean'], std=params['std'])
        x = torch.from_numpy(framed_imgs[0])

        if use_cuda:
            x = x.cuda(gpu)
            if use_float16:
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()

        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        features, regression, classification, anchors = model(x)

        preds = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, nms_threshold)
        
        if not preds:
            continue

        preds = invert_affine(framed_metas, preds)[0]

        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    'image_name': image_name,
                    # 'category_id': label + 1,
                    'category_id': label + 5,
                    'score': float(score),
                    'bbox': box.tolist(),
                }

                results.append(image_result)

    
    return results

def eval_forward(model, img_dir, coco = None, image_ids = None, threshold=0.05, set_name = None):
    results = []

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        img_filepath = os.path.join(img_dir, image_info['file_name'])
        ori_imgs, framed_imgs, framed_metas = preprocess(img_filepath, max_size=input_sizes[compound_coef], mean=params['mean'], std=params['std'])
        x = torch.from_numpy(framed_imgs[0])

        if use_cuda:
            x = x.cuda(gpu)
            if use_float16:
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()

        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        features, regression, classification, anchors = model(x)

        preds = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, nms_threshold)
        
        if not preds:
            continue

        preds = invert_affine(framed_metas, preds)[0]

        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    'image_id': image_id,
                    # 'category_id': label + 1,
                    'category_id': label + 5,
                    'score': float(score),
                    'bbox': box.tolist(),
                }

                results.append(image_result)
    
    return results



def calculate_metric(model, anno_filepath, img_dir, split = None, output_dir = None):
    max_images = 40

    coco_gt = COCO(anno_filepath)
    image_ids = coco_gt.getImgIds()
    # image_ids = image_ids[:max_images]

    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model.cuda(gpu)
        if use_float16:
            model.half()

    pre_results = eval_forward(model, img_dir, coco_gt, image_ids)

    # json_filepath = os.path.join(output_dir, f'det_results.json') 
    # if os.path.exists(json_filepath):
    #     os.remove(json_filepath)
    # json.dump(pre_results, open(json_filepath, 'w'), indent=4)
    # coco_pred = coco_gt.loadRes(json_filepath)

    coco_pred = coco_gt.loadRes(pre_results)
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    mean_ap = coco_eval.stats[0].item()  # stats[0] records AP@[0.5:0.95]
    print(f'AP@[0.5:0.95] = {mean_ap}')

    return mean_ap

    
if __name__ == '__main__':
    # SET_NAME = params['val_set']
    # VAL_GT = f'datasets/{params["project_name"]}/annotations/instances_{SET_NAME}.json'
    # VAL_IMGS = f'datasets/{params["project_name"]}/{SET_NAME}/'
    GT_ = '/data2/hongrui/project/dataset/annotation/erosiveulcer_fine/cancer/test.json'
    IMGS_ = '/data2/zzhang/annotation/erosiveulcer_fine/train/images/'
    
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                    ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
    # model.module.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    weights_paths = [#'logs/erosiveulcer_fine/2022-12-02_23-07-20/weight/efficientdet-d2_best.pth.tar',
                    #  'logs/erosiveulcer_fine/2022-12-02_23-07-42/weight/efficientdet-d2.pth.tar',
                     'logs/erosiveulcer_fine/2022-11-29_11-04-17/weight/efficientdet-d2_best.pth.tar',
                     'logs/erosiveulcer_fine/2022-11-29_11-19-30/weight/efficientdet-d2_best.pth.tar',
                     ]
                     
    for weights_path in weights_paths:
        print(f'loading weight from {weights_path}')
        log_dict['weight'] = weights_path
        output_dir = weights_path[:weights_path.find('weight')]
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
        if isinstance(model, CustomDataParallel):
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        mean_ap = calculate_metric(model, GT_, IMGS_)

        log_dict['AP@[0.5:0.95]'] = mean_ap
        current_time = get_current_time()
        result_filepath = os.path.join(output_dir, f'parameters.txt') 
        # if os.path.exists(result_filepath):
        #     os.remove(result_filepath)
        log_file = open(result_filepath, "a+")
        log_file.write('\n')
        log_file.write('\n')
        log_file.write('----------------------Eval-------------------------' + '\n')
        log_file.write(current_time + '\n')

        p = vars(args)
        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.write('\n')

        for key, val in log_dict.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.write('----------------------Eval END-------------------------' + '\n')

        log_file.write('\n')
        log_file.close()
        print('\n')