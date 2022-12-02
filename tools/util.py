import sys
import time
import os
import numpy as np
import random
import shutil
import cv2
import scipy

from PIL import Image
import math
from math import pi
import imageio
import pytz
import datetime
import csv
from matplotlib import pyplot as plt
import torch
import json
import base64
import matplotlib.patches as patches
from matplotlib.path import Path

rgb_palette = [[0, 0, 0], [0, 255, 0], [0, 0, 255],  [255, 0, 0],
                       [128, 50, 0], [0, 128, 50], [128, 128, 0],
                       [90, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128],[128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],]
                       
rgb_palette = [item for sublist in rgb_palette for item in sublist]
zero_pad = 256 * 3 - len(rgb_palette)
for i in range(zero_pad):
    rgb_palette.append(0)

def colorize_mask_to_bgr(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(rgb_palette)
    mask_rgb = new_mask.convert("RGB")
    mask_rgb = np.array(mask_rgb)
    mask_bgr = mask_rgb[:,:,::-1]
    return mask_bgr

def write_list_to_txt(txt_filepath, lists):
    f = open(txt_filepath, "a+")
    for l in lists:
        f.write(str(l)+'\n')
    f.close()#

def write_dict_to_txt(txt_filepath, dicts):
    label_k = dicts.keys()
    # label_v = dicts.values()
    f = open(txt_filepath, "a+")
    for k in label_k:
        f.write(f'{k}: {str(dicts[k])}'+'\n')
    f.close()#

def read_txt_to_list(txt_filepath):
    lists = []
    with open(txt_filepath) as f:
        lines = f.readlines()
        for line in lines:
            lists.append(line.strip('\n'))
    # print(len(lists))
    return lists

def sort_left_right_lane(label):
    if len(np.unique(label)) < 3: #delete the image with only one rail and no rail at all.
        # print('np.unique(label)', np.unique(label))
        return None, None
    left_0 = min(label[label>0])
    # print(label[label>left_0].min())
    right_0 = 2*left_0
    # print('left_0 ',left_0, 'right_0 ', right_0)
    return label*(label==left_0), label*(label==right_0)

def find_bottom_lane_location_in_labels(label):
    left_lane, right_lane = sort_left_right_lane(label.copy())
    if not isinstance(left_lane, np.ndarray):
        return -1, -1
    # cv2.imwrite('left_lane.png', left_lane)
    # cv2.imwrite('right_lane.png', right_lane)
    assert label.ndim == 2
    h, w = label.shape
    max_y_left = left_lane.nonzero()[0].max()
    max_y_right = right_lane.nonzero()[0].max()
    # print('max_y_left, max_y_right', max_y_left, max_y_right)
    max_y = min(max_y_left, max_y_right)
    left_x_pos = left_lane[max_y-20:max_y+1].nonzero()[1].mean()
    right_x_pos = right_lane[max_y-20:max_y+1].nonzero()[1].mean()
    # print('left_x_pos, right_x_pos', left_x_pos, right_x_pos)
    # print(left_lane[max_y-20:max_y].nonzero())
    # print(right_lane[max_y-20:max_y].nonzero())
    return left_x_pos, right_x_pos


def plot_and_save_complex_func(res,  mask_name = None, out_img_filepath = None, text_str = None, debug = False):
    if debug:
        print(f'call {sys._getframe().f_code.co_name}')
    if 1 <= len(res) <= 3:
        row = 1
        col = len(res)

    elif 4 <= len(res) <= 8:
        row = 2
        col = len(res)//2 + len(res)%2

    elif 9 <= len(res) <= 12:
        # col = math.sqrt(len(res)) if math.sqrt(len(res))%1 == 0 else int(math.sqrt(len(res))) + 1
        # row = int(math.sqrt(len(res)))
        row = 3
        col = len(res)//3 + len(res)%3

    else:
        row = len(res)//5 + 1
        col = 5
        
    
    for i in range(len(res)):
        # img = res[i].astype(np.uint8)
        img = res[i]
        if img.ndim == 2:
            height, width = img.shape
        elif img.ndim == 3:
            height, width, _ = img.shape
        else:
            return
        if i == 0:
            ori_img = img.copy()
            
            ax = plt.subplot(row, col, i+1), plt.imshow(img), plt.title(mask_name[i]), plt.xticks([]), plt.yticks([])
            if text_str:
                # ax.text(2.0, 9.5, text_str, fontsize=10)
                # ax.text(.05, .95, text_str, color = 'red', transform=ax.transAxes, ha="left", va="top")
                plt.text(.05, .95, text_str, fontsize = 6, color = 'red', ha = "left", va = "top", rotation = 0, wrap = True)

        else:
            if mask_name:
                if 'fitted_curves' in mask_name[i]:
                    # ax = plt.subplot(row, col, i+1), plt.imshow(np.zeros((height,width))), plt.title(mask_name[i]), plt.xticks([]), plt.yticks([])
                    ax = plt.subplot(row, col, i+1), plt.imshow(ori_img), plt.title(mask_name[i]), plt.xticks([]), plt.yticks([])
                    x_plot, curve_fits = img
                    lw = 0.2
                    for j in range(len(curve_fits)):
                        plt.plot(np.polyval(curve_fits[j], x_plot), x_plot, color='lightgreen', linestyle='--', linewidth=lw)
                    plt.xlim(0, width)
                    plt.ylim(height, 0)
                    
                else:
                    ax = plt.subplot(row, col, i+1), plt.imshow(img), plt.title(mask_name[i]), plt.xticks([]), plt.yticks([])
            else:    
                ax = plt.subplot(row, col, i+1), plt.imshow(img), plt.title(f'res_{i}'), plt.xticks([]), plt.yticks([])
            if text_str:
                # ax.text(2.0, 9.5, text_str, fontsize=10)
                # ax.text(.05, .95, text_str, color = 'red', transform=ax.transAxes, ha="left", va="top")
                plt.text(.05, .95, text_str, fontsize = 6, color = 'red', ha = "left", va = "top", rotation = 0, wrap = True)

        if isinstance(res[i], np.ndarray) and res[i].ndim == 2:
            plt.gray()

    plt.subplots_adjust(wspace=0)
    if out_img_filepath:
        print(f'saving {out_img_filepath}')
        figure = plt.gcf()  
        figure.set_size_inches(16, 9)
        plt.savefig(out_img_filepath, dpi=900, bbox_inches='tight')
        plt.close()
    # else:
    #     plt.show()
    #     time.sleep(3)
    #     plt.close()
    print()



def compose_img_label_pre(img, label, pre):
    img = img.astype(np.uint8)
    label = label.astype(np.uint8)
    pre = pre.astype(np.uint8)
    # img = put_text_on_image(img, 'image')

    label_bgr = colorize_mask_to_bgr(label.copy())
    # label_bgr = put_text_on_image(label_bgr, 'color gt')

    cat_img_label = np.concatenate((img, label_bgr), axis=1)

    pre_bgr = colorize_mask_to_bgr(pre.copy())
    # pre_bgr = put_text_on_image(pre_bgr, 'color pre')

    composed = 0.7*img + 0.3*pre_bgr
    # composed = put_text_on_image(composed, 'img+pre')

    cat_img_pre = np.concatenate((composed, pre_bgr), axis=1)

    out_img = np.concatenate((cat_img_label, cat_img_pre), axis=0)
    return out_img


def morphologyEx_open(image, kernel_size = 3, debug = False):
    if debug:
        print(f'call {sys._getframe().f_code.co_name}')
        start = time.time()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size, kernel_size))
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    if debug:
        print(f'end of {sys._getframe().f_code.co_name}, costs {round(time.time()-start, 2)} s')

    return opening

def multiple_img_with_binary_mask(img, mask):
    if img.ndim == 3:
        for i in range(3):
            img[:,:,i] *= mask
    elif img.ndim == 2:
        img *= mask

    return img

def fill_img_with_gaussian_noise_based_on_mask(img, gaussian_mask, gaussian_noise):
    for c in range(3):
        img[:,:,c] = img[:,:,c]*(gaussian_mask==0)
        gaussian_noise[:,:,c] = gaussian_noise[:,:,c]*(gaussian_mask>0)
    img = img + gaussian_noise

    return img

def select_img_pixel_on_binary_mask(img, mask):
    if img.ndim == 3:
        for i in range(3):
            img[:,:,i][mask] = 0
    elif img.ndim == 2:
        img[mask] = 0 

    return img


def morphologyEx_close(image, kernel_size = 3, debug = False):
    if debug:
        print(f'call {sys._getframe().f_code.co_name}')
        start = time.time()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size, kernel_size))
    opening = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    if debug:
        print(f'end of {sys._getframe().f_code.co_name}, costs {round(time.time()-start, 2)} s')

    return opening


def gasuss_noise(image):
    img = np.random.randint(0, 255, (image.shape))
    return img

def write_line_to_csv(csvlogfile, row, head = None):
    with open(csvlogfile, 'a+', newline='') as csvfile:
        # fieldnames = ['first_name', 'last_name']
        # writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # writer.writeheader()
        # writer.writerow({'first_name': 'Baked', 'last_name': 'Beans'})
        # writer.writerow({'first_name': 'Lovely', 'last_name': 'Spam'})
        # writer.writerow({'first_name': 'Wonderful', 'last_name': 'Spam'})
        writer = csv.writer(csvfile, dialect='excel')
        if head is not None:
            writer.writerow(head)
        writer.writerow(row)


def write_list_to_row_in_csv(csvlogfile, row):
    with open(csvlogfile, 'a+', newline='') as csvfile:
        # fieldnames = ['first_name', 'last_name']
        # writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # writer.writeheader()
        # writer.writerow({'first_name': 'Baked', 'last_name': 'Beans'})
        # writer.writerow({'first_name': 'Lovely', 'last_name': 'Spam'})
        # writer.writerow({'first_name': 'Wonderful', 'last_name': 'Spam'})
        writer = csv.writer(csvfile, dialect='excel')
        writer.writerow(row)

def get_plot_sample(image, seg_pre, seg_mask = None, decision=None, is_pos = None, seg_loss_mask = None,  blur=False):
        
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(seg_pre, torch.Tensor):
        seg_pre = seg_pre.detach().cpu().numpy()
    
    if not seg_mask is None and isinstance(seg_mask, torch.Tensor):
        seg_mask = seg_mask.detach().cpu().numpy()
        
    if not seg_loss_mask is None and isinstance(seg_loss_mask, torch.Tensor):
        seg_loss_mask = seg_loss_mask.detach().cpu().numpy()
    
    if not decision is None and isinstance(decision, torch.Tensor):
        decision = decision.detach().cpu().numpy()
        
    if not is_pos is None and isinstance(is_pos, torch.Tensor):
        is_pos = is_pos.detach().cpu().numpy()
    # print('image.shape', image.shape)
    
    image = np.squeeze(image)
    
    # print('seg_mask', seg_mask)
    image = (image - image.min())/(image.max() - image.min())
    
    # print('1 seg_pre[1].shape', seg_pre[1].shape) #seg_pre[1].shape (600, 600)
    
    seg_pre_1ch = np.squeeze(seg_pre[1])

    # print('1 image.shape', image.shape) #image.shape (3, 600, 600)
    # print('1 seg_pre.shape', seg_pre.shape) #seg_pre.shape (2, 600, 600)
    # print('1 seg_pre_1ch.shape', seg_pre_1ch.shape) #seg_pre_1ch.shape (600, 600)

    n_col = 3
    if not seg_mask is None:
        n_col += 1
    elif not seg_loss_mask is None:
        n_col += 1
    else:
        pass
    
    ## figure = plt.figure(figsize=(1,n_col)) 
    figure = plt.figure(figsize=(4*n_col, 5))
    
    
    pos = 1
    plt.subplot(1, n_col, pos)
    plt.xticks([])
    plt.yticks([])
    if not is_pos is None:
        is_pos = np.squeeze(is_pos)
        plt.title(f'InputImage\n{is_pos}', verticalalignment = 'bottom', fontsize = 'small')
    else:
        plt.title('InputImage', verticalalignment = 'bottom', fontsize = 'small')
    # plt.ylabel('Input image', multialignment='center')
    if image.ndim == 3:
        image = np.transpose(image, axes=[1, 2, 0])
    if image.shape[0] < image.shape[1]:
        trans_flag = True
        if image.ndim == 3:
            image = np.transpose(image, axes=[1, 0, 2])
        else:
            image = np.transpose(image)
        
        if seg_pre.ndim == 3:
            seg_pre = np.transpose(seg_pre, axes=[1, 0, 2])
        else:
            seg_pre = np.transpose(seg_pre)

        seg_pre_1ch = np.transpose(seg_pre_1ch)
    else:
        trans_flag = False

    seg_argmax = np.argmax(seg_pre, axis=0)
    # print('2 seg_argmax.shape', seg_argmax.shape) #seg_argmax.shape (600, 600)
    # print('2 seg_pre.shape', seg_pre.shape)
    # print('2 seg_pre_1ch.shape', seg_pre_1ch.shape)
    # print('image.shape', image.shape)
    
    if image.ndim == 2:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)
    pos += 1

    if not seg_mask is None:
        seg_mask = np.squeeze(seg_mask)
        label = seg_mask.copy()
        label = np.transpose(label) if trans_flag else label
        label_min = label.min()
        label_max = label.max()
        plt.subplot(1, n_col, pos)
        plt.xticks([])
        plt.yticks([])
        # plt.title('Groundtruth')
        # print('label.shape', label.shape)
        plt.title(f'segMask\n{label_min:.2f}->{label_max:.2f}', verticalalignment = 'bottom', fontsize = 'small')
        plt.imshow(label, cmap="gray")
        pos += 1

    if not seg_loss_mask is None:
        seg_loss_mask = np.squeeze(seg_loss_mask)
        label = seg_loss_mask.copy()
        label = np.transpose(label) if trans_flag else label
        label_min = label.min()
        label_max = label.max()
        plt.subplot(1, n_col, pos)
        plt.xticks([])
        plt.yticks([])
        # plt.title('Groundtruth')
        plt.title(f'segLossMask\n{label_min:.2f}->{label_max:.2f}', verticalalignment = 'bottom', fontsize = 'small')
        plt.imshow(label, cmap="gray")
        pos += 1

    plt.subplot(1, n_col, pos)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'OutputScaled\nscope:{seg_pre_1ch.min():.2f}->{seg_pre_1ch.max():.2f}', verticalalignment = 'bottom', fontsize = 'small')
    # plt.ylabel('OutputScaled', multialignment='center')
    if blur:
        normed = seg_pre_1ch / seg_pre_1ch.max()
        blured = cv2.blur(normed, (32, 32))
        plt.imshow((blured / blured.max() * 255).astype(np.uint8), cmap="jet")
    else:
        plt.imshow((seg_pre_1ch / seg_pre_1ch.max() * 255).astype(np.uint8), cmap="jet")
    pos += 1
    
    plt.subplot(1, n_col, pos)
    plt.xticks([])
    plt.yticks([])
    if decision is None:
        plt.title('OutputArgmax', verticalalignment = 'bottom', fontsize = 'small')
        # plt.ylabel('Output', multialignment='center')
    else:
        decision = np.squeeze(decision)
        plt.title(f"OutputArgmax\nConf:{decision:.2f}", verticalalignment = 'bottom', fontsize = 'small')
        # plt.ylabel(f"Output:{decision:.2f}", multialignment='center')
    # display max
    vmax_value = max(1, np.max(seg_pre))
    plt.imshow(seg_argmax, cmap="jet", vmax=vmax_value)
    

    
    # plt.show()

    return figure

def save_tsv_file(input_img_dir, tsv_file_name, output_dir):
    out_tsv_filepath = os.path.join(output_dir, tsv_file_name)
    if os.path.exists(out_tsv_filepath):
        os.remove(out_tsv_filepath)


    img_names = os.listdir(input_img_dir)
    # img_names
    img_names = sorted(img_names)
    # print(img_names)
    label_info = {"objects": [{"class": "pot"}], "metadata": ""}
    with open(out_tsv_filepath, "a") as tf:
        for i, img_name in enumerate(img_names):
            print(f'{i}/{len(img_names)}, {img_name}')
            img_filepath = os.path.join(input_img_dir, img_name)
            with open(img_filepath, "rb") as f:
                im_b64_str = base64.b64encode(f.read()).decode()
                data = "\t".join([str(i), json.dumps(label_info), im_b64_str])
                tf.writelines(data + "\n")

            # img_filepath = 'file://' + os.path.join(input_img_dir, img_name)
            # data = "\t".join([str(i), json.dumps(label_info), img_filepath])
            # tf.writelines(data + "\n")

def get_random_shape(edge_num=3, ratio=0.7, width=400, height=300):
    '''
      There is the initial point and 3 points per cubic bezier curve. 
      Thus, the curve will only pass though n points, which will be the sharp edges.
      The other 2 modify the shape of the bezier curve.
      edge_num, Number of possibly sharp edges
      points_num, number of points in the Path
      ratio, (0, 1) magnitude of the perturbation from the unit circle, 
    '''
    wid = np.random.randint(100, 500)
    hei = np.random.randint(100, 500)
    points_num = edge_num*3 + 1
    angles = np.linspace(0, 2*np.pi, points_num)
    codes = np.full(points_num, Path.CURVE4)
    codes[0] = Path.MOVETO
    # Using this instad of Path.CLOSEPOLY avoids an innecessary straight line
    verts = np.stack((np.cos(angles), np.sin(angles))).T * \
        (2*ratio*np.random.random(points_num)+1-ratio)[:, None]
    verts[-1, :] = verts[0, :]
    path = Path(verts, codes)
    # draw paths into images
    fig = plt.figure()
    ax = fig.add_subplot(111)
    patch = patches.PathPatch(path, facecolor='black', lw=2)
    ax.add_patch(patch)
    ax.set_xlim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax.set_ylim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax.axis('off')  # removes the axis to leave only the shape
    fig.canvas.draw()
    # convert plt images into numpy images
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape((fig.canvas.get_width_height()[::-1] + (3,)))
    plt.close(fig)
    # postprocess
    data = cv2.resize(data, (wid, hei))[:, :, 0]
    data = (1 - np.array(data > 0).astype(np.uint8))*255
    # corrdinates = np.where(data > 0)
    # xmin, xmax, ymin, ymax = np.min(corrdinates[0]), np.max(
    #     corrdinates[0]), np.min(corrdinates[1]), np.max(corrdinates[1])
    #region = Image.fromarray(data).crop((ymin, xmin, ymax, xmax))
    # data = data[xmin:xmax, ymin:ymax]

    if random.random() < 0.5:
        data = np.rot90(data, 1)
    # data = np.rot90(data, 1)

    return data


def put_text_on_image(image, text):
    h, w, _ = image.shape
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # org
    y0 = 70
    x0 = 10
    # org = (10, y0)
    
    # fontScale
    if h > 1500 or w > 1500:
        fontScale = 3
        thickness = 2
    elif 1500 >= h > 1000 or 1500 >= w > 1000:
        fontScale = 2
        thickness = 2
    else:
        fontScale = 2
        thickness = 2

    if image.ndim == 3:
        # white color in BGR
        color = (255, 255, 255)
    else:
        color = 255
    
    # Line thickness of 2 px
    # thickness = 2
    dy = 70
    # Using cv2.putText() method
    for i, txt in enumerate(text.split('\n')):
        y = y0+i*dy
        image = cv2.putText(image, txt, (x0, y), font, 
                        fontScale, color, thickness, cv2.LINE_AA)
    return image
    
def mosaic_img_based_on_mask(img, mosaic_mask):
    noise = np.random.randint(0, 255, (img.shape))
    for c in range(3):
        img[:,:,c] = img[:,:,c]*(mosaic_mask<=0)
        noise[:,:,c] = noise[:,:,c]*(mosaic_mask>0)
    img = img + noise

    return img

def mask_to_boxes_corners(mask):
    '''
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    (xmin, ymin, xmax, ymax) in absolute floating points coordinates.
    The coordinates in range [0, width or height].
    '''
    
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)





if __name__ == '__main__':
    # read_txt_to_list('/home/hongrui/project/metro_pro/edge_detection/chdis_v2.txt')
    pass