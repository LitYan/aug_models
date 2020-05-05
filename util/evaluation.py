import numpy as np
import gala.evaluate as ev
from skimage.measure import label
import time
#---------------for evaluation----------------------

def get_figure_of_merit(pred, mask, const_index=0.1):  # _calculate_by_neighbor_point
    """
    针对真值图 mask 和预测图pred计算F值并返回
    针对pred中的每个点，遍历其60邻域内最近的mask点,计算d_i,最终计算F_score
    本方法速度最慢
    :param pred: 预测图，[0,255]，背景为0，前景为255
    :param mask: 真值图，[0,255]，背景为0，前景为255
    :return: f_score
    """
    assert np.shape(pred) == np.shape(mask)
    num_pred = np.count_nonzero(pred[pred == 255])
    num_mask = np.count_nonzero(mask[mask == 255])
    num_max = num_pred if num_pred > num_mask else num_mask

    temp = 0.0
    for index_x in range(0, pred.shape[0]):
        for index_y in range(0, pred.shape[1]):
            if pred[index_x, index_y] == 255:
                distance = get_dis_from_mask_point(mask, index_x, index_y)
                temp = temp + 1 / (1 + const_index * pow(distance, 2))
    f_score = (1.0 / num_max) * temp
    return f_score

def get_dis_from_mask_point(mask, index_x, index_y, neighbor_length=60):
    """
    计算检测到的边缘点与离它最近边缘点的距离
    """

    if mask[index_x, index_y] == 255:
        return 0
    distance = neighbor_length / 2
    region_start_row = 0
    region_start_col = 0
    region_end_row = mask.shape[0]
    region_end_col = mask.shape[1]
    if index_x - neighbor_length > 0:
        region_start_row = index_x - neighbor_length
    if index_x + neighbor_length < mask.shape[0]:
        region_end_row = index_x + neighbor_length
    if index_y - neighbor_length > 0:
        region_start_col = index_y - neighbor_length
    if index_y + neighbor_length < mask.shape[1]:
        region_end_col = index_y + neighbor_length
    # Get the corrdinate of mask in neighbor region
    # becuase the corrdinate will be chaneged after slice operation, we add it manually
    x, y = np.where(mask[region_start_row: region_end_row, region_start_col: region_end_col] == 255)
    if len(x) == 0:
        min_distance = 30
    else:
        min_distance = np.amin(
            np.linalg.norm(np.array([x + region_start_row - index_x, y + region_start_col - index_y]), axis=0))

    return min_distance

def calculate_vi_ri_ari(result, gt):
    # false merges(缺失), false splits（划痕）
    merger_error, split_error = ev.split_vi(result, gt)
    vi = merger_error + split_error
    ri = ev.rand_index(result, gt)
    adjust_ri = ev.adj_rand_index(result, gt)
    return {'vi':vi,'ri':ri,'adjust_ri':adjust_ri,
            'merger_error':merger_error,
            'split_error':split_error}

def get_pixel_accuracy(pred,mask):
    assert np.shape(pred)==np.shape(mask)
    t=np.sum(pred==mask)
    s=np.size(mask)
    return t/s
def calculate_ap(label_pred, num_pred, label_mask, num_mask):
    thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    tp = np.zeros(10)
    count=0
    m_iou=0
    for i_pred in range(1, num_pred + 1):
        intersect_mask_labels = list(np.unique(label_mask[label_pred == i_pred]))  # 获得与之相交的所有label
        if 0 in intersect_mask_labels:
            intersect_mask_labels.remove(0)

        if len(intersect_mask_labels) == 0:  # 如果pred的某一个label没有与之对应的mask的label,则继续下一个label
            continue
        intersect_mask_label_area = np.zeros((len(intersect_mask_labels), 1))
        union_mask_label_area = np.zeros((len(intersect_mask_labels), 1))

        for index, i_mask in enumerate(intersect_mask_labels):
            intersect_mask_label_area[index, 0] = np.count_nonzero(label_pred[label_mask == i_mask] == i_pred)
            union_mask_label_area[index, 0] = np.count_nonzero((label_mask == i_mask) | (label_pred == i_pred))

        iou = intersect_mask_label_area / union_mask_label_area
        max_iou = np.max(iou, axis=0)
        m_iou+=max_iou
        count+=1
        tp[thresholds < max_iou] = tp[thresholds < max_iou] + 1
    fp = num_pred - tp
    fn = num_mask - tp
    map_score = np.average(tp / (tp + fp + fn))
    m_iou/=count
    return {'map_score':map_score,'m_iou':m_iou}

def get_map_miou_vi_ri_ari(pred, mask,boundary=255):
    """
    map https://www.kaggle.com/c/data-science-bowl-2018/overview/evaluation
    :param pred: 预测图，[0, 255], 前景为255， 背景为0
    :param mask: 真值图，[0, 255]，前景为255， 背景为0
    :return: map F VI RI aRI
    """
    assert np.shape(pred) == np.shape(mask)
    # 1px边缘闭合
    pred[[0,-1], :] = boundary
    pred[:,[0,-1]] = boundary
    mask[[0,-1], :] = boundary
    mask[:, [0,-1]] = boundary
    time1 = time.time()
    label_mask, num_mask = label(mask, neighbors=4, background=boundary, return_num=True)
    label_pred, num_pred = label(pred, neighbors=4, background=boundary, return_num=True)
    time2 = time.time()
#     print('labeling cost:', time2 - time1)

    results={}
#     time1 = time.time()
#     f = get_figure_of_merit(pred, mask)
#     time2 = time.time()
#     print('f:', f, ' cal f cost:', time2 - time1)
#     results['f']=f

#     time1 = time.time()
    m_ap_iou = calculate_ap(label_pred, num_pred, label_mask, num_mask)
#     time2 = time.time()
#     print('map:', m_ap_iou['map_score'],' miou:',m_ap_iou['m_iou'], ' cal map cost:', time2 - time1)
    results.update(m_ap_iou)

#     time1 = time.time()
    vi_ri = calculate_vi_ri_ari(label_pred, label_mask)
#     time2 = time.time()
#     print('vi:', vi_ri['vi'], ' ri:', vi_ri['ri'], ' cal vi_ri_ari cost:', time2 - time1)
    results.update(vi_ri)
    return results

import glob
import os
import cv2

def evaluate(pred_path,mask_path,boundary=255):
    l=glob.glob(pred_path+'/*.png')
    count=0
    eval_results={}
    with open(pred_path+'_eval.txt','w') as log:
        now = time.strftime('%c')
        log.write('=============Evaluation (%s)=============\n' % now)
    for f in l:
        name=os.path.basename(f)
        pred=cv2.imread(f,0)
        mask=cv2.imread(os.path.join(mask_path,name),0)
        eval_result=get_map_miou_f_vi_ri_ari(pred,mask,boundary=boundary)
        count+=1
        message=str(count)+':\t'
        for k,v in eval_result.items():
            if k in eval_results:
                eval_results[k]+=v
            else:
                eval_results[k]=v
            message+='%s: %.5f\t'%(k,v)
        print(message)
        message+='\n'
        with open(pred_path + '_eval.txt', 'a') as log:
            log.write(message)
    message='total %d:\n'%count
    for k,v in eval_results.items():
        message += 'm_%s: %.5f\t' % (k, v/count)
    print(message)
    message += '\n'
    with open(pred_path + '_eval.txt', 'a') as log:
        log.write(message)    