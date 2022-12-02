import json
import os
import datetime
import cv2
import numpy as np
from scipy import ndimage as nd
import imageio as io

import pycocotools
from pycocotools.mask import encode
import pycocotools.coco as coco
from pycocotools.coco import COCO
import random
from PIL import Image
from matplotlib import pyplot as plt
from util import *

def convert_json():
    annFile = '/data2/zzhang/annotation/erosiveulcer_fine/trainfp0927.json'
    image_dir = '/data2/zzhang/annotation/erosiveulcer_fine/train/images/'

    output_fg_dir = 'data/ori_fg'
    output_bg_dir = 'data/ori_bg'

    if not os.path.exists(output_fg_dir):
        os.makedirs(output_fg_dir)

    if not os.path.exists(output_bg_dir):
        os.makedirs(output_bg_dir)
    #### 0:erosive 1:ulcer 2:others 3:hemorrhage, 4:cancer

    coco=COCO(annFile)

    imgInfos = coco.imgs
    # print('imgInfos', imgInfos)
    # idx = np.random.randint(0,len(imgInfos))
    print(f'total num is {len(imgInfos)}')
    idx_lists = [i for i in range(len(imgInfos))]

    random.shuffle(idx_lists)
    # print('idx_lists', idx_lists)
    cnt = 0
    target_num = 100
    for i, idx in enumerate(idx_lists):
        # for idx in range(len(imgInfos)):
        # idx = np.random.randint(0,len(imgInfos))
        imgInfo  = imgInfos[idx]
        img_name = imgInfo['file_name']
        img_filepath = os.path.join(image_dir, img_name)
        print(f'processing {i}, {cnt}/{target_num}, id: {idx}, {img_name}')
        if not os.path.exists(img_filepath):
            continue
        # image = np.array(Image.open(img_filepath))
        image = cv2.imread(img_filepath)
        # plt.imshow(image, interpolation='nearest')
        # plt.show()

        # plt.imshow(image)
        cat_ids = coco.getCatIds()
        
        # print('cat_ids', cat_ids)
        try:
            anns_ids = coco.getAnnIds(imgIds=imgInfo['id'], catIds=cat_ids, iscrowd=None)
            # print('anns_ids', anns_ids)
            anns  = coco.loadAnns(anns_ids)
            # print('anns ', anns )

            # coco.showAnns(anns)

            h, w, _ = image.shape
            mask = np.zeros((h,w), dtype=np.uint8)
            # print('mask.shape', mask.shape, np.unique(mask))
            fg_flag = False
            for j in range(len(anns)):
                # print('anns [i]', anns[i])
                keys = anns[j].keys()
                if not 'segmentation' in keys:
                    break
                seg_point_list = anns[j]['segmentation']
                if len(seg_point_list) <= 0:
                    break
                cls_id = anns[j]['category_id']
                mask += coco.annToMask(anns[j]) * cls_id
                if cls_id == 5:
                    fg_flag = True
            # if len(np.unique(mask)) <= 1:
            #     continue
            # print('mask.shape', mask.shape, np.unique(mask))
            col_mask = colorize_mask_to_bgr(mask)
            compose_img = 0.8 * image + 0.2 * col_mask
            compose_img = compose_img.astype(np.uint8)
            if fg_flag:
                output_dir = output_fg_dir
            else:
                output_dir = output_bg_dir
            img_name_prefix = img_name[:-4]
            out_img_filepath = os.path.join(output_dir, img_name)
            out_label_filepath = os.path.join(output_dir, img_name_prefix + '.png')
            

            cv2.imwrite(out_img_filepath, image)
            cv2.imwrite(out_label_filepath, mask)
            
            out_col_filepath = os.path.join(output_dir, img_name_prefix + '_col.jpg')
            out_comp_filepath = os.path.join(output_dir, img_name_prefix + '_comp.jpg')

            cv2.imwrite(out_col_filepath, col_mask)
            cv2.imwrite(out_comp_filepath, compose_img)

            cnt += 1
            if cnt > 100:
                return
        except:
            print(f'    processing {idx}, {img_name} failed')
        plt.imshow(mask)



def convert_json():
    # annFile = '/data2/zzhang/annotation/erosiveulcer_fine/trainfp0927.json'
    # image_dir = '/data2/zzhang/annotation/erosiveulcer_fine/train/images/'

    output_fg_dir = 'data/ori_fg'
    output_bg_dir = 'data/ori_bg'

    if not os.path.exists(output_fg_dir):
        os.makedirs(output_fg_dir)

    if not os.path.exists(output_bg_dir):
        os.makedirs(output_bg_dir)
    #### 0:erosive 1:ulcer 2:others 3:hemorrhage, 4:cancer
    ## coco label +1

    coco=COCO(annFile)

    imgInfos = coco.imgs
    # print('imgInfos', imgInfos)
    # idx = np.random.randint(0,len(imgInfos))
    print(f'total num is {len(imgInfos)}')
    idx_lists = [i for i in range(len(imgInfos))]

    random.shuffle(idx_lists)
    # print('idx_lists', idx_lists)
    cnt = 0
    target_num = 100
    for i, idx in enumerate(idx_lists):
        # for idx in range(len(imgInfos)):
        # idx = np.random.randint(0,len(imgInfos))
        imgInfo  = imgInfos[idx]
        img_name = imgInfo['file_name']
        img_filepath = os.path.join(image_dir, img_name)
        print(f'processing {i}, {cnt}/{target_num}, id: {idx}, {img_name}')
        if not os.path.exists(img_filepath):
            continue
        # image = np.array(Image.open(img_filepath))
        image = cv2.imread(img_filepath)
        # plt.imshow(image, interpolation='nearest')
        # plt.show()

        # plt.imshow(image)
        cat_ids = coco.getCatIds()
        
        # print('cat_ids', cat_ids)
        try:
            anns_ids = coco.getAnnIds(imgIds=imgInfo['id'], catIds=cat_ids, iscrowd=None)
            # print('anns_ids', anns_ids)
            anns  = coco.loadAnns(anns_ids)
            # print('anns ', anns )

            # coco.showAnns(anns)

            h, w, _ = image.shape
            mask = np.zeros((h,w), dtype=np.uint8)
            # print('mask.shape', mask.shape, np.unique(mask))
            fg_flag = False
            for j in range(len(anns)):
                # print('anns [i]', anns[i])
                keys = anns[j].keys()
                if not 'segmentation' in keys:
                    break
                seg_point_list = anns[j]['segmentation']
                if len(seg_point_list) <= 0:
                    break
                cls_id = anns[j]['category_id']
                mask += coco.annToMask(anns[j]) * cls_id
                if cls_id == 5:
                    fg_flag = True
            # if len(np.unique(mask)) <= 1:
            #     continue
            # print('mask.shape', mask.shape, np.unique(mask))
            col_mask = colorize_mask_to_bgr(mask)
            compose_img = 0.8 * image + 0.2 * col_mask
            compose_img = compose_img.astype(np.uint8)
            if fg_flag:
                output_dir = output_fg_dir
            else:
                output_dir = output_bg_dir
            img_name_prefix = img_name[:-4]
            out_img_filepath = os.path.join(output_dir, img_name)
            out_label_filepath = os.path.join(output_dir, img_name_prefix + '.png')
            

            cv2.imwrite(out_img_filepath, image)
            cv2.imwrite(out_label_filepath, mask)
            
            out_col_filepath = os.path.join(output_dir, img_name_prefix + '_col.jpg')
            out_comp_filepath = os.path.join(output_dir, img_name_prefix + '_comp.jpg')

            cv2.imwrite(out_col_filepath, col_mask)
            cv2.imwrite(out_comp_filepath, compose_img)

            cnt += 1
            if cnt > 100:
                return
        except:
            print(f'    processing {idx}, {img_name} failed')
        plt.imshow(mask)

def cal_cnt():
    


    annFile = '/data2/hongrui/project/dataset/annotation/erosiveulcer_fine/trainfp1108.json'
    # annFile = '/data3/zzhang/annotation/erosiveulcer_fine/trainfp1108.json' ## matched images number is 7609
    image_dir = '/data2/zzhang/annotation/erosiveulcer_fine/train/images/' #image number is 13280
    # with bbox : ##[4659, 1640, 1612, 785, 2736]
    # with bbox and segmentation:[523, 566, 0, 544, 2736] 
    
    # annFile = '/data2/zzhang/annotation/erosiveulcer_fine/test0928.json' #### matched images number is 2036
    # image_dir = '/data2/zzhang/annotation/erosiveulcer_fine/train/images/'
    # with bbox : [2234, 782, 514, 0, 0]
    # with bbox and segmentation: [95, 11, 514, 0, 0]

    # annFile = '/data2/zzhang/annotation/erosiveulcer_fine/trainfp0927.json'  # matched images number is 6696
    # image_dir = '/data2/zzhang/annotation/erosiveulcer_fine/train/images/'
    # with bbox : ##[4580, 1246, 1612, 785, 1762]
    # with bbox and segmentation: [508, 172, 0, 544, 1762]

    # annFile = '/data2/dechunwang/dataset/cleaned_data_annotation/adenomatous/train.json'
    # annFile = '/data2/dechunwang/dataset/gastric_object_detection/erosive_annotations/train.json'
    # annFile = '/data2/dechunwang/dataset/gastric_object_detection/erosiveulcer/trainfp0214.json'
    # annFile = '/data2/dechunwang/dataset/gastric_object_detection/erosiveulcer/trainfp0214.json'
    annFile = '/data2/dechunwang/dataset/coco/anno/instances_train2014.json'
    annFile = '/data2/dechunwang/dataset/gastric_object_detection/erosive_annotations/test_mix.json'
    annFile = '/data2/dechunwang/dataset/gastric_object_detection/erosive_annotations/test.json'
    annFile = '/data2/dechunwang/dataset/cleaned_data_annotation/adenomatous/train.json'
    annFile = '/data2/zzhang/annotation/erosiveulcer_fine/testfp1006.json'
    annFile = '/data2/zzhang/annotation/erosiveulcer_fine/test0928.json'
    
    image_dir = '/data2/zzhang/annotation/erosiveulcer_fine/train/images/'


    ##
    #### 0:erosive 1:ulcer 2:others 3:hemorrhage, 4:cancer
    ## coco label +1
    
    img_names = os.listdir(image_dir)
    print(f'total image num is {len(img_names)}')

    coco=COCO(annFile)

    imgIds = coco.getImgIds()

    # print('imgInfos', imgInfos)
    # print('imgIds', imgIds)
    # idx = np.random.randint(0,len(imgInfos))
    print(f'total num is {len(imgIds)}')
    cat_ids = coco.getCatIds()
    print(f'lenght of cat_ids is {len(cat_ids)}, {cat_ids}')
    # return
    # idx_lists = [i for i in range(len(imgInfos))]
    num_lists = [0 for _ in range(len(cat_ids))]
    pair_cnt = 0
    random.shuffle(imgIds)
    for i, img_id in enumerate(imgIds):
        # idx = np.random.randint(0,len(imgInfos))
        
        # idx = np.random.randint(0,len(imgInfos))
        # print(imgInfo)
        imgInfo = coco.loadImgs(img_id)[0]
        # print('imgInfo', imgInfo)

        img_name = imgInfo['file_name']
        # img_filepath = os.path.join(image_dir, img_name)
        # if not os.path.exists(img_filepath):
        #     continue
        pair_cnt += 1
        # print(f'processing {i}/{len(imgIds)}, pair:{pair_cnt}, {img_name}')
        cat_ids = coco.getCatIds()
        # print('cat_ids', cat_ids)
        # annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)

        # coco_annotations = self.coco.loadAnns(annotations_ids)
        # print("imgInfo['id']", type(imgInfo['id']))
        try:
            anns_ids = coco.getAnnIds(imgIds=imgInfo['id'], catIds=cat_ids, iscrowd=None)

            # print('anns_ids', anns_ids)
            anns  = coco.loadAnns(anns_ids)
            # print('anns ', anns )
            # coco.showAnns(anns)
            for j in range(len(anns)):
                # print('anns [i]', anns[i])
                keys = anns[j].keys()
                if not 'bbox' in keys:
                    continue
                seg_point_list = anns[j]['segmentation']
                if len(seg_point_list) <= 0:
                    continue
                cls_id = anns[j]['category_id']
                pose_index = cat_ids.index(cls_id)
                # if cls_id == 1 or cls_id == 2:
                #     cnt += 1
                #     break
                if cls_id in cat_ids:
                    num_lists[pose_index] += 1
        except:
            # print('error')
            pass
        # break
    # print(cnt, avi_cnt)
    print(num_lists)

def count_json_annos():
    out_anno_dir = '/data2/hongrui/project/dataset/annotation/erosiveulcer_fine/cancer'
    json_files = ['train.json', 'val.json', 'test.json']
    
    
    for json_file in json_files:
        annFile = os.path.join(out_anno_dir, json_file)
        coco=COCO(annFile)
        imgIds = coco.getImgIds()
        print(f'total num is {len(imgIds)}')


# def load_all(self, anno_file, shuffle):
#     """
#     initialize all entries given annotation json file

#     Parameters:
#     ----------
#     anno_file: str
#         annotation json file
#     shuffle: bool
#         whether to shuffle image list
#     """
#     image_set_index = []
#     labels = []
#     coco = COCO(anno_file)
#     img_ids = coco.getImgIds()
#     for img_id in img_ids:
#         # filename
#         image_info = coco.loadImgs(img_id)[0]
#         filename = image_info["file_name"]
#         subdir = filename.split('_')[1]
#         height = image_info["height"]
#         width = image_info["width"]
#         # label
#         anno_ids = coco.getAnnIds(imgIds=img_id)
#         annos = coco.loadAnns(anno_ids)
#         label = []
#         for anno in annos:
#             cat_id = int(anno["category_id"])
#             bbox = anno["bbox"]
#             assert len(bbox) == 4
#             xmin = float(bbox[0]) / width
#             ymin = float(bbox[1]) / height
#             xmax = xmin + float(bbox[2]) / width
#             ymax = ymin + float(bbox[3]) / height
#             label.append([cat_id, xmin, ymin, xmax, ymax, 0])
#         if label:
#             labels.append(np.array(label))
#             image_set_index.append(os.path.join(subdir, filename))

#     if shuffle:
#         import random
#         indices = range(len(image_set_index))
#         random.shuffle(indices)
#         image_set_index = [image_set_index[i] for i in indices]
#         labels = [labels[i] for i in indices]
#     # store the results
#     self.image_set_index = image_set_index
#     self.labels = labels


def rewrite_json():
    annFile = '/data3/zzhang/annotation/erosiveulcer_fine/trainfp1123.json'
    image_dir = '/data2/zzhang/annotation/erosiveulcer_fine/train/images/'
    ##[304, 551, 0, 487, 2717]

    # annFile = '/data2/zzhang/annotation/erosiveulcer_fine/trainfp0927.json'
    # image_dir = '/data2/zzhang/annotation/erosiveulcer_fine/train/images/'
    ##[290, 159, 0, 487, 1762]
    #### 0:erosive 1:ulcer 2:others 3:hemorrhage, 4:cancer
    ## coco label +1
    

    out_anno_dir = '/data2/hongrui/project/dataset/annotation/erosiveulcer_fine/cancer'
    if not os.path.exists(out_anno_dir):
        os.makedirs(out_anno_dir)
    # json_file = 'trainfp1108.json'
    # json_files = ['train.json', 'val.json', 'test.json']
    # splits = [0.6, 0.2, 0.2]
    out_json_file = os.path.join(out_anno_dir, 'trainfp1123.json')
    if os.path.exists(out_json_file):
        os.remove(out_json_file)
    

    img_idx = 1
    annot_id = 1
    images = []
    annotations = []
    
    
    
    coco=COCO(annFile)

    imgIds = coco.getImgIds()

    # print('imgInfos', imgInfos)
    # idx = np.random.randint(0,len(imgInfos))
    print(f'total num is {len(imgIds)}')
    cat_ids = coco.getCatIds()
    print('cat_ids', cat_ids)
    random.shuffle(imgIds)
    for i, img_id in enumerate(imgIds):
        flag = False
        imgInfo = coco.loadImgs(img_id)[0]
        img_name = imgInfo['file_name']
        img_filepath = os.path.join(image_dir, img_name)
        
        if not os.path.exists(img_filepath):
            continue

        print(f'processing {i}/{len(imgIds)}, {img_name}')
        
        cat_ids = coco.getCatIds()
        # print('cat_ids', cat_ids)

        anns_ids = coco.getAnnIds(imgIds=imgInfo['id'], catIds=cat_ids, iscrowd=None)
        # print('anns_ids', anns_ids)
        anns  = coco.loadAnns(anns_ids)
        # print('anns ', anns )

        # coco.showAnns(anns)
        
        for j in range(len(anns)):
            # print(f'{i}, anns[{j}]', anns[j])
            keys = anns[j].keys()
            if not 'segmentation' in keys:
                # break
                continue
            seg_point_list = anns[j]['segmentation']
            if len(seg_point_list) <= 0:
                # break
                continue
            cls_id = anns[j]['category_id']

            if cls_id == 5:
                flag = True
                annotations.append({"segmentation" : anns[j]['segmentation'],
                            "area" : np.float(anns[j]['area']),
                            "iscrowd" : 0,
                            "image_id" : img_idx,
                            "bbox" : anns[j]['bbox'],
                            "category_id" : anns[j]['category_id'],
                            "id": annot_id})
                annot_id += 1
        if flag:
            images.append({"date_captured" : "2016",
                        "file_name" : img_name, # remove "/"
                        "id" : img_idx,
                        "license" : 1,
                        "url" : "",
                        "height" : int(imgInfo['height']),
                        "width" : int(imgInfo['width'])})
            
            img_idx += 1
            flag = False
    # for i in range(len(images)):
    print(f'matched images number is {img_idx - 1}')
    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year='2022',
            contributor=None,
            date_created='2022',
        ),
        licenses=[dict(url=None, id=0, name=None,)],
        # images=[
        #     # license, url, file_name, height, width, date_captured, id
        # ],
        images=images,
        annotations=annotations,
        type="instances",
        # annotations=[
        #     # segmentation, area, iscrowd, image_id, bbox, category_id, id
        # ],
        categories=[
            # supercategory, id, name
        ],
        img_dir="/data2/zzhang/annotation/erosiveulcer_fine/train/images/",
    )  
    labels = {0:'erosive', 1:'ulcer', 2:'others', 3:'hemorrhage', 4:'cancer'}
    ids = labels.keys()
    for i in ids:
        data["categories"].append(
                dict(supercategory=None, id=i+1, name=labels[i],)
            )
        # break
    with open(out_json_file, "w") as f:
        json.dump(data, f)


    
def split_json():
    annFile = '/data2/hongrui/project/dataset/annotation/erosiveulcer_fine/cancer/trainfp1123.json'
    image_dir = '/data2/zzhang/annotation/erosiveulcer_fine/train/images/'
    

    out_anno_dir = '/data2/hongrui/project/dataset/annotation/erosiveulcer_fine/cancer'
    if not os.path.exists(out_anno_dir):
        os.makedirs(out_anno_dir)
    # json_file = 'trainfp1108.json'
    
    
    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year='2022',
            contributor=None,
            date_created='2022',
        ),
        licenses=[dict(url=None, id=0, name=None,)],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        # images=images,
        # annotations=annotations,
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
        img_dir="/data2/zzhang/annotation/erosiveulcer_fine/train/images/",
    )  
    labels = {0:'erosive', 1:'ulcer', 2:'others', 3:'hemorrhage', 4:'cancer'}
    ids = labels.keys()
    for i in ids:
        data["categories"].append(
                dict(supercategory=None, id=i+1, name=labels[i],)
            )
    
    coco=COCO(annFile)

    imgIds = coco.getImgIds()

    # print('imgInfos', imgInfos)
    # idx = np.random.randint(0,len(imgInfos))
    print(f'total num is {len(imgIds)}')
    cat_ids = coco.getCatIds()
    print('cat_ids', cat_ids)
    random.shuffle(imgIds)
    img_idx = 1
    annot_id = 1

    
    all_annotations = [ [] for _ in range(3)]
    all_images = [ [] for _ in range(3)]
    json_files = ['train.json', 'val.json', 'test.json']
    
    for i, img_id in enumerate(imgIds):
        if i < 0.6*len(imgIds):
            annotations = all_annotations[0]
            images = all_images[0]
        elif i < 0.8*len(imgIds):
            annotations = all_annotations[1]
            images = all_images[1]
        else:
            annotations = all_annotations[2]
            images = all_images[2]
        flag = False
        imgInfo = coco.loadImgs(img_id)[0]
        img_name = imgInfo['file_name']
        img_filepath = os.path.join(image_dir, img_name)
        
        if not os.path.exists(img_filepath):
            continue

        print(f'processing {i}/{len(imgIds)}, {img_name}')
        
        cat_ids = coco.getCatIds()
        # print('cat_ids', cat_ids)

        anns_ids = coco.getAnnIds(imgIds=imgInfo['id'], catIds=cat_ids, iscrowd=None)
        # print('anns_ids', anns_ids)
        anns  = coco.loadAnns(anns_ids)
        # print('anns ', anns )

        # coco.showAnns(anns)
        
        for j in range(len(anns)):
            # print(f'{i}, anns[{j}]', anns[j])
            keys = anns[j].keys()
            if not 'segmentation' in keys:
                # break
                continue
            seg_point_list = anns[j]['segmentation']
            if len(seg_point_list) <= 0:
                # break
                continue
            cls_id = anns[j]['category_id']

            if cls_id == 5:
                flag = True
                annotations.append({"segmentation" : anns[j]['segmentation'],
                            "area" : np.float(anns[j]['area']),
                            "iscrowd" : 0,
                            "image_id" : img_idx,
                            "bbox" : anns[j]['bbox'],
                            "category_id" : anns[j]['category_id'],
                            "id": annot_id})
                annot_id += 1
        if flag:
            images.append({"date_captured" : "2016",
                        "file_name" : img_name, # remove "/"
                        "id" : img_idx,
                        "license" : 1,
                        "url" : "",
                        "height" : int(imgInfo['height']),
                        "width" : int(imgInfo['width'])})
            
            img_idx += 1
            flag = False
    # for i in range(len(images)):
    print(f'matched images number is {img_idx - 1}')
    

    for i in range(3):
        data['annotations'] = all_annotations[i]
        data['images'] = all_images[i]
        out_json_file = os.path.join(out_anno_dir, json_files[i])
        if os.path.exists(out_json_file):
            os.remove(out_json_file)
        with open(out_json_file, "w") as f:
            json.dump(data, f)


if __name__ == "__main__":
    # copy_paste_generate_dataset()
    # rewrite_json()
    # cal_cnt()
    # split_json()
    count_json_annos()