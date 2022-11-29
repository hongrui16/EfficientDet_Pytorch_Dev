import os
import torch
import numpy as np
from skimage.measure import label, regionprops, find_contours

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2
from colorsys import rgb_to_hsv, hsv_to_rgb
from skimage.color import rgb2hsv, hsv2rgb


""" Convert a mask to border image """
def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border

""" Mask to bounding boxes """
def mask_to_bbox(mask):
    bboxes = []
    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])

    return bboxes

def parse_mask(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask

class CocoDataset(Dataset):
    def __init__(self, root_dir, set='train2017', transform=None):

        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):

        # load class names (name -> label)
        self.cat_ids = self.coco.getCatIds()
        # print('self.cat_ids', self.cat_ids)
        categories = self.coco.loadCats(self.cat_ids)
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations



class FakeCocoDataset(Dataset):
    def __init__(self, root_dir = None, set='train2017', transform=None, args = None, **kwargs):
        self.args = args
        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform

        self.img_dir = kwargs.get('img_dir', None)
        self.annFilePath = kwargs.get('annFile', None)
        split = kwargs.get('split', 'train')
        
        self.cancer_label_id = 5
        self.use_paste_aug = args.use_paste_aug if split == 'train' else False
        self.coco = COCO(self.annFilePath)
        self.image_ids = self.coco.getImgIds()
        print(f'{split} example number is {len(self.image_ids)}')
        # self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        # categories = self.coco.loadCats(self.coco.getCatIds())
        cat_ids = self.coco.getCatIds()
        # print('self.cat_ids', cat_ids)
        # return
        categories = self.coco.loadCats(cat_ids)
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key
        print('self.labels', self.labels)
        # print('...............')
        # print('...............')
        # print('...............')
        # print('...............')
        
        # print('...............')
        # print('...............')
        # print('...............')


    def __len__(self):
        return len(self.image_ids)



    def __getitem__(self, idx):
        image_index = idx
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        # path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        path = os.path.join(self.img_dir, image_info['file_name'])
        if not os.path.exists(path):
            while True:
                image_index = np.random.randint(0, len(self.image_ids))
                image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
                # path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
                path = os.path.join(self.img_dir, image_info['file_name'])
                if os.path.exists(path):
                    break
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            sample = {'img': img, 'annot': annotations}
            if self.transform:
                sample = self.transform(sample)
            return sample

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        h, w, _ = img.shape
        mask = np.zeros((h,w), dtype=np.uint8)
        for idx, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue
            if self.args.call_cancer_only and a['category_id'] != self.cancer_label_id:
                continue
            annotation = np.zeros((1, 5))
            
                # annotation[0, :4] = a['bbox']
            cls_id = a['category_id'] 
            try:
                mask += self.coco.annToMask(a) * cls_id
            except:
                # print('error')
                continue
            
            annotation[0, 4] = cls_id 
            annotation[0, :4] = a['bbox']
            annotations = np.append(annotations, annotation, axis=0)

        # if len(mask[mask==self.cancer_label_id]) > 0 and self.args.use_paste_aug and np.random.rand() < 0.5:
        if len(mask[mask==self.cancer_label_id]) > 0 and self.args.use_paste_aug:
            if len(mask.shape) == 2 and len(img.shape) == 3:
                annotation = np.zeros((1, 5))
                t_img, _, t_bbox = paste_instance_on_the_same_image_syncs(img.copy(), mask.copy(), img.copy(), mask.copy(), self.cancer_label_id)
                if (not t_img is None) and (not t_bbox is None):
                    annotation[0, :4] = np.array(t_bbox)
                    annotation[0, 4] = self.cancer_label_id
                    annotations = np.append(annotations, annotation, axis=0)
                    img = t_img
        if self.args.call_cancer_only:
            annotations[:, 4][annotations[:, 4] != self.cancer_label_id] = 0 
            annotations[:, 4][annotations[:, 4] == self.cancer_label_id] = 0 
        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]


        sample = {'img': img, 'annot': annotations}
        if self.transform:
            sample = self.transform(sample)
        return sample

    # def load_image(self, image_index):
    #     image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
    #     path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
    #     img = cv2.imread(path)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #     return img.astype(np.float32) / 255.

    # def load_annotations(self, image_index):
    #     # get ground truth annotations
    #     annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
    #     annotations = np.zeros((0, 5))

    #     # some images appear to miss annotations
    #     if len(annotations_ids) == 0:
    #         return annotations

    #     # parse annotations
    #     coco_annotations = self.coco.loadAnns(annotations_ids)
    #     for idx, a in enumerate(coco_annotations):

    #         # some annotations have basically no width / height, skip them
    #         if a['bbox'][2] < 1 or a['bbox'][3] < 1:
    #             continue

    #         annotation = np.zeros((1, 5))
    #         annotation[0, :4] = a['bbox']
    #         annotation[0, 4] = a['category_id'] - 1
    #         annotations = np.append(annotations, annotation, axis=0)

    #     # transform from [x, y, w, h] to [x1, y1, x2, y2]
    #     annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
    #     annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

    #     return annotations



def mask_to_bbox_corners(mask, mode='XYXY'):
    '''given a binary mask (0 or int>0) returns the
    bounding box as tuple row0, row1, col0, col1 
    if mode=='XYXY' return a list row0, col0, row1,col1

    Enum of different ways to represent a box.

    In detectron2:
    XYXY_ABS= 0
    (xmin, ymin, xmax, ymax) in absolute floating points coordinates.
    The coordinates in range [0, width or height].
    '''
    # print('     mask_to_bbox_corners.......mask', mask.shape)
    col_0 = np.nonzero(mask.any(axis=0))[0][0]
    col_1 = np.nonzero(mask.any(axis=0))[0][-1]
    row_0 = np.nonzero(mask.any(axis=1))[0][0]
    row_1 = np.nonzero(mask.any(axis=1))[0][-1]
    
    if mode == 'XYXY':
        xmin = int(col_0)
        xmax = int(col_1)
        ymin = int(row_0)
        ymax = int(row_1)
        return [xmin, ymin, xmax, ymax]


def paste_instance_on_the_same_image_syncs(fg_img, fg_label, bg_img, bg_label, target_id):
    # fg_mask_filepath = fg_img_filepath.replace('.jpg', '.png')
    # bg_masl_filepath = bg_img_filepath.replace('.jpg', '.png')
    # print('fg_img ', fg_img.shape)
    # print('fg_label ', fg_label.shape)
    safe_margin = 3
    bg_h, bg_w, _ = bg_img.shape
    fg_mask = fg_label
    fg_mask = fg_mask==target_id
    fg_mask = fg_mask.astype(np.uint8)
    fg_bbox = mask_to_bbox_corners(fg_mask)
    xmin, ymin, xmax, ymax = fg_bbox
    fg_h, fg_w = fg_mask.shape
    if len(fg_mask[fg_mask == 1]) /(fg_h * fg_w) > 0.4:
        return None, None, None
    b_h = ymax - ymin
    b_w = xmax - xmin
    if b_w * b_h / (fg_h * fg_w) > 0.4:
        return None, None, None
    # print('fg_bbox', fg_bbox)

    # print('fg_mask', fg_mask.nonzero())
    # print('fg_mask', np.unique(fg_mask))
    # fg_bboxes = mask_to_bbox(fg_mask)
    # cv2.imwrite('tep.jpg', 255*fg_mask)
    # print('fg_bboxes', fg_bboxes)
    # xmin, ymin, xmax, ymax = fg_bboxes[0]
    fg_box_h = ymax - ymin
    fg_box_w = xmax - xmin
    
    scopes = [1/15, 5/15]
    min_w = int(scopes[0]*bg_w)
    min_h = int(scopes[0]*bg_h)

    max_w = int(scopes[1]*bg_w)
    max_h = int(scopes[1]*bg_h)

    new_fg_box_h = np.random.randint(min_h, max_h)
    new_fg_box_w = np.random.randint(min_w, max_w)

    ratio_h = new_fg_box_h/fg_box_h
    ratio_w = new_fg_box_w/fg_box_w

    fg_h, fg_w = fg_mask.shape
    new_fg_h = int(ratio_h*fg_h)
    new_fg_w = int(ratio_w*fg_w)

    fg_img = cv2.resize(fg_img, (new_fg_w, new_fg_h))
    fg_mask = cv2.resize(fg_mask, (new_fg_w, new_fg_h), interpolation = cv2.INTER_NEAREST)

    overlap_pixel = 15
    # start_x = int(ratio_w*xmin) - new_fg_box_w//2 if int(ratio_w*xmin) - new_fg_box_w//2 > 0 else 0
    # end_x = int(ratio_w*xmax) + new_fg_box_w//2 if int(ratio_w*xmax) + new_fg_box_w//2 < new_fg_w else new_fg_w
    # start_y = int(ratio_h*ymin) - new_fg_box_h//2 if int(ratio_h*ymin) - new_fg_box_h//2 > 0 else 0
    # end_y = int(ratio_h*ymax) + new_fg_box_h//2 if int(ratio_h*ymax) + new_fg_box_h//2 < new_fg_h else new_fg_h    
    start_x = int(ratio_w*xmin) - overlap_pixel if int(ratio_w*xmin) - overlap_pixel > 0 else 0
    end_x = int(ratio_w*xmax) + overlap_pixel if int(ratio_w*xmax) + overlap_pixel < new_fg_w else new_fg_w
    start_y = int(ratio_h*ymin) - overlap_pixel if int(ratio_h*ymin) - overlap_pixel > 0 else 0
    end_y = int(ratio_h*ymax) + overlap_pixel if int(ratio_h*ymax) + overlap_pixel < new_fg_h else new_fg_h
    # print(start_y, end_y, start_x, end_x)
    fg_block_img = fg_img[start_y : end_y, start_x : end_x]
    fg_block_mask = fg_mask[start_y : end_y, start_x : end_x]
    
    fg_block_ycbcr = rgb2hsv(fg_block_img[::-1])
    fg_block_lightness = fg_block_ycbcr[:,:,2]
    fg_block_lightness = cv2.GaussianBlur(fg_block_lightness, (7,7), 2)
    mean_fg_block_lightness = fg_block_lightness.mean()

    bg_img_ycbcr = rgb2hsv(bg_img[::-1])
    bg_img_lightness = bg_img_ycbcr[:,:,2]
    bg_img_lightness = cv2.GaussianBlur(bg_img_lightness, (5,5), 2)
    hue_offset = 0.3
    available_paste_mask = (bg_img_lightness< mean_fg_block_lightness + hue_offset) * (bg_img_lightness>mean_fg_block_lightness - hue_offset)

    binary_fg_block = cv2.cvtColor(fg_block_img, cv2.COLOR_BGR2GRAY)
    binary_fg_block = cv2.GaussianBlur(binary_fg_block, (7,7), 2)
    mean_value_bin_fg_block = binary_fg_block.mean()

    binary_bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
    binary_bg_img = cv2.GaussianBlur(binary_bg_img, (5,5), 2)
    binary_offset = 40
    binary_mask = (binary_bg_img > mean_value_bin_fg_block - binary_offset) * (binary_bg_img < mean_value_bin_fg_block + binary_offset)

    new_fg_img = np.zeros(bg_img.shape, dtype=np.uint8)
    new_fg_mask = np.zeros(bg_label.shape, dtype=np.uint8)

    fg_block_h = fg_block_mask.shape[0]
    fg_block_w = fg_block_mask.shape[1]
    # offset_h = bg_h - fg_block_h
    # offset_w = bg_w - fg_block_w
    # print(f'fg_block_h {fg_block_h}; bg_h {bg_h}; offset_h {offset_h}')
    target_mask = bg_label>0
    target_mask = target_mask.astype(np.uint8)
    bg_bbox = mask_to_bbox_corners(target_mask)
    bg_xmin, bg_ymin, bg_xmax, bg_ymax = bg_bbox
    # bg_bboxes = mask_to_bbox(target_mask)
    # bg_xmin, bg_ymin, bg_xmax, bg_ymax = bg_bboxes[0]
    ext_bg_xmin = bg_xmin - new_fg_box_w //2 if bg_xmin - new_fg_box_w //2 > 0 else 0
    ext_bg_ymin = bg_ymin - new_fg_box_h //2 if bg_ymin - new_fg_box_h //2 > 0 else 0
    ext_bg_xmax = bg_xmax + new_fg_box_w //2 if bg_xmax + new_fg_box_w //2 < bg_w else bg_w
    ext_bg_ymax = bg_ymax + new_fg_box_h //2 if bg_ymax + new_fg_box_h //2 < bg_h else bg_h
    
    bg_label_temp = bg_label.copy()
    # 
    bg_label_temp[ext_bg_ymin:ext_bg_ymax, ext_bg_xmin:ext_bg_xmax] = 1

    bg_label_temp[0:fg_block_h //2 + 1, :] = 1
    bg_label_temp[:, 0:fg_block_w//2+1] = 1
    bg_label_temp[-fg_block_h //2 -1:, :] = 1
    bg_label_temp[:, -fg_block_w//2 + 1:] = 1
    # cv2.imwrite('temp0.jpg', 255*bg_label_temp)
    available_paste_mask = (bg_label_temp<=0)*available_paste_mask*binary_mask
    nonzeros = np.nonzero(available_paste_mask)
    nonzero_y = nonzeros[0].tolist()
    nonzero_x = nonzeros[1].tolist()
    # max_loop_counter = 10000
    if len(nonzero_y) > 0:
        rd_index = np.random.randint(0, len(nonzero_y))
        c_x = nonzero_x[rd_index]
        c_y = nonzero_y[rd_index]
            # if fg_block_w//2 + 1 < c_x < bg_w - fg_block_w//2 - 1 and fg_block_h//2 + 1 < c_y < bg_h - fg_block_h//2 -1:
            #     break
    else:
        c_y = np.random.randint(fg_block_h//2 + 1, bg_h - fg_block_h//2 -1)
        c_x = np.random.randint(fg_block_w//2 + 1, bg_w - fg_block_w//2 - 1)
    new_start_x = c_x - fg_block_w//2
    new_start_y = c_y - fg_block_h//2
    # print(f'fg_block_img.shape {fg_block_img.shape}')
    # print(f'new_fg_img[new_start_y:new_start_y+fg_block_h, new_start_x:new_start_x+fg_block_w] {new_fg_img[new_start_y:new_start_y+fg_block_h, new_start_x:new_start_x+fg_block_w].shape}')
    # print(f'new_fg_img.shape {new_fg_img.shape} new_start_x:new_start_x+fg_block_w = {new_start_x}: {new_start_x+fg_block_w}')
    # print(f'new_fg_img.shape {new_fg_img.shape} new_start_y:new_start_y+fg_block_h = {new_start_y}: {new_start_y+fg_block_h}')
    # print(new_start_y+fg_block_h -new_start_y, new_start_x+fg_block_w - new_start_x)
    new_fg_img[new_start_y:new_start_y+fg_block_h, new_start_x:new_start_x+fg_block_w] = fg_block_img
    new_fg_mask[new_start_y + safe_margin:new_start_y+fg_block_h-safe_margin,
                 new_start_x+safe_margin:new_start_x+fg_block_w-safe_margin] = fg_block_mask[safe_margin:-safe_margin, safe_margin:-safe_margin]

    
    new_fg_mask[new_fg_mask > 0] = 1
    new_fg_mask = new_fg_mask.astype(np.float64)

    # neg_mask = fg_mask == False
    # fg_mask = np.dstack([fg_mask]*3)
    # neg_mask = np.dstack([neg_mask]*3)

    # # compose_img = fg_img[fg_mask] + bg_img[neg_mask]
    # compose_img = fg_img*fg_mask + bg_img*neg_mask

    
    # print(np.unique(fg_mask))
    # print(fg_mask.shape)
    new_fg_mask = cv2.GaussianBlur(new_fg_mask, (21,21), 2)
    new_fg_label = new_fg_mask.copy()
    new_fg_label[new_fg_label>0.6] = 1
    new_fg_label[new_fg_label<=0.4] = 0
    
    temp_new_fg_mask = new_fg_mask.copy()
    

    bg_mask = 1-new_fg_mask
    # foreground = cv2.multiply(fg_mask, fg_img)
    # background = cv2.multiply(neg_mask, bg_img)
    # compose_img = cv2.add(foreground, background)
    new_fg_mask = np.dstack([new_fg_mask]*3)
    bg_mask = np.dstack([bg_mask]*3)
    compose_img = new_fg_img*new_fg_mask + bg_img*bg_mask

    compose_img = compose_img.astype(np.uint8) #合成的图像

    temp_new_fg_mask = temp_new_fg_mask>0
    out_box = mask_to_bbox_corners(temp_new_fg_mask.astype(np.uint8))
    b_xmin, b_ymin, b_xmax, b_ymax = out_box
    # out_boxes = mask_to_bbox(temp_new_fg_mask.astype(np.uint8))
    # b_xmin, b_ymin, b_xmax, b_ymax = out_boxes[0]
    # bg_b_ycbcr = rgb2ycbcr(bg_img[b_ymin:b_ymax, b_xmin:b_xmax][::-1])
    # comp_b_ycbcr = rgb2ycbcr(compose_img[b_ymin:b_ymax, b_xmin:b_xmax][::-1])
    # comp_b_ycbcr[:,:,0] = bg_b_ycbcr[:,:,0]

    bg_b_ycbcr = rgb2hsv(bg_img[b_ymin:b_ymax, b_xmin:b_xmax][::-1])
    comp_b_ycbcr = rgb2hsv(compose_img[b_ymin:b_ymax, b_xmin:b_xmax][::-1])
    comp_b_ycbcr[:,:,2] = 0.75*bg_b_ycbcr[:,:,2] +  0.25*comp_b_ycbcr[:,:,2]

    # new_block = ycbcr2rgb(comp_b_ycbcr)
    new_block = hsv2rgb(comp_b_ycbcr)
    new_block *= 255
    new_block = new_block.astype(np.uint8)
    compose_img[b_ymin:b_ymax, b_xmin:b_xmax] = new_block[::-1]

    bg_label[new_fg_label>0] = 0
    compose_label = new_fg_label + bg_label
    compose_label = compose_label.astype(np.uint8) #合成的mask
    out_bbox = [b_xmin, b_ymin, b_xmax - b_xmin, b_ymax - b_ymin]
    return compose_img, compose_label, out_bbox

def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32)/255. - self.mean) / self.std), 'annot': annots}
