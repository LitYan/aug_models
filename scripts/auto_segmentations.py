import cv2
from skimage import morphology
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import argparse
import time
from util.util import mkdir,save_image
from util.evaluation import get_map_miou_vi_ri_ari

####灰度图分割#####
def plot_img(img,seg,title):
    print(seg.dtype,np.unique(seg))
    plt.figure(figsize=(20,20))
    plt.subplot(121), plt.imshow(img, 'gray'), plt.title('input')
    plt.axis('off')
    plt.subplot(122), plt.imshow(seg, 'gray'), plt.title(title)
    plt.axis('off')
    plt.show()
def seg_kmeans_gray(img,flag=cv2.KMEANS_RANDOM_CENTERS):
    '''
    cv2.kmeans(data, K, bestLabels, criteria, attempts, flags)

    参数：
        data: 分类数据，最好是np.float32的数据，每个特征放一列。
        K: 分类数，opencv2的kmeans分类是需要已知分类数的。
        bestLabels：预设的分类标签或者None
        criteria：迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
            其中，type有如下模式：
             —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
             —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
             —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
        attempts：重复试验kmeans算法次数，将会返回最好的一次结果
        flags：初始中心选择，有两种方法：
            ——cv2.KMEANS_PP_CENTERS;
            ——cv2.KMEANS_RANDOM_CENTERS

    返回值：
        compactness：紧密度，返回每个点到相应重心的距离的平方和
        labels：结果标记，每个成员被标记为0,1等
        centers：由聚类的中心组成的数组

    '''
 
    # 展平
    img_flat = img.reshape((img.shape[0] * img.shape[1], 1))
    img_flat = np.float32(img_flat)
 
    # 迭代参数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 20, 0.5)

    # 进行聚类
    compactness, labels, centers = cv2.kmeans(img_flat, 2, None, criteria, 10, flag)
 
    # 显示结果
    labels = 1-labels if 2*np.count_nonzero(labels)>len(labels) else labels
    img_output = labels.reshape((img.shape[0], img.shape[1]))
    img_output = np.uint8(img_output)*255
#     plot_img(img,img_output,'kmeans')
    return img_output

def seg_otsu(img):
    ret_OTSU, th_OTSU = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # Otsu 滤波
#     plot_img(img,th_OTSU,'OTSU')
    return ret_OTSU,th_OTSU

def seg_otsu_inverse(img):
    ret_OTSU,th_OTSU = seg_otsu(img)
    th_OTSU = 255 - th_OTSU
    return th_OTSU

def seg_thres(img,threshold=0.5):
    th = cv2.threshold(input, threshold*255, 255, cv2.THRESH_BINARY)
#     plot_img(img,th,'thres_'+str(threshold))
    return th

def seg_adathres(img,type=cv2.ADAPTIVE_THRESH_MEAN_C):
    '''
    type:cv2.ADAPTIVE_THRESH_GAUSSIAN_C/cv2.ADAPTIVE_THRESH_MEAN_C
    '''
    th_ADAPTIVE = cv2.adaptiveThreshold(img, 255,type, cv2.THRESH_BINARY, 5, 2)
#     plot_img(img,th_ADAPTIVE,'ADAPTIVE')
    return th_ADAPTIVE

def watershed(img):
    _,th = seg_otsu(img)
    kernel=np.ones((3,3),np.uint8)
    opening=cv2.morphologyEx(th,cv2.MORPH_OPEN,kernel,iterations=2) #开运算(open)：先腐蚀后膨胀,去除孤立点
    sure_bg=cv2.dilate(opening,kernel,iterations=3)
    dist_transfrom=cv2.distanceTransform(opening,cv2.DIST_L2 ,5)
    ret,sure_fg=cv2.threshold(dist_transfrom,0.2*dist_transfrom.max(),255,0)

    sure_fg=np.uint8(sure_fg)
    unknow=cv2.subtract(sure_bg,sure_fg) #背景-前景
    ret,marker=cv2.connectedComponents(sure_fg)
    marker=marker+1
    marker[unknow==255]=0    
    img_bgr = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    marker = cv2.watershed(img_bgr,marker)
#     print(np.unique(marker),marker.dtype)
    marker[marker != -1] = 0
    marker[marker == -1] = 255
    marker=np.uint8(marker)
#     plot_img(img,marker,'watersheld')
    return marker
def edge(img,type='sobel'):
    img = cv2.GaussianBlur(img,(3,3),0)
#     Sobel_x_or_y = cv2.Sobel(src, ddepth, dx, dy, dst, ksize, scale, delta, borderType)
#     参数：
#     第一个参数是需要处理的图像；
#     第二个参数是图像的深度，-1表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度；
#     dx和dy表示的是求导的阶数，0表示这个方向上没有求导，一般为0、1、2。
    if type == 'sobel':
        sobel_x = cv2.Sobel(img,-1, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img,-1, 0, 1, ksize=3)
        out = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    elif type == 'scharr':
        scharr_x = cv2.Scharr(img,-1,1,0)
        scharr_y = cv2.Scharr(img,-1,0,1)
        out = cv2.addWeighted(scharr_x,0.5,scharr_y,0.5,0)
    elif type == 'laplac':
        out = cv2.Laplacian(img,-1)
#     plot_img(img,out,type)
    return out

def canny(img):
    out = cv2.Canny(img, 50,150)
    return out
def post_process(output):
    result = morphology.closing(output,morphology.square(3))
    result = morphology.skeletonize(output/255)*255
    result = np.uint8(result)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path',type=str,required=True)
    parser.add_argument('--labels_path',type=str,required=True)
    parser.add_argument('--results_dir',type=str,default='./results/others')
    parser.add_argument('--method',type=str,default='otsu',choices=['otsu','kmeans','watershed','canny'])
    parser.add_argument('--boundary',type=int,default=255)
    opt = parser.parse_args()
    save_dir = os.path.join(opt.results_dir,opt.method)
    mkdir(save_dir)
    imgs = sorted(glob.glob(opt.images_path+'/*.png'))
    methods={'otsu':'seg_otsu_inverse',
            'canny':'canny',
            'kmeans':'seg_kmeans_gray',
            'watershed':'watershed'}
    method = eval(methods[opt.method])
    #do otsu|canny|watershed|kmeans...
    count=0
    eval_results = {}
    with open(save_dir+'_eval.txt','w') as log:
        now = time.strftime('%c')
        log.write('=============Evaluation (%s)=============\n' % now)
    for img in imgs:
        input = cv2.imread(img,0)
        name = os.path.basename(img).split('.')[0]
        seg = method(input)
        seg = post_process(seg)
#         plot_img(input,seg,opt.method)
#         break
        save_path = os.path.join(save_dir,name+'.png')
        save_image(seg,save_path)
        #eval 
        eval_start = time.time()
        count+=1
        mask = cv2.imread(os.path.join(opt.labels_path,name+'.png'),0)
        eval_result=get_map_miou_vi_ri_ari(seg,mask,boundary=opt.boundary)
        message='%04d: %s \t'%(count,name)
        for k,v in eval_result.items():
            if k in eval_results:
                eval_results[k]+=v
            else:
                eval_results[k]=v
            message+='%s: %.5f\t'%(k,v)
            
        print(message,'cost: %.4f'%(time.time()-eval_start))
        with open(save_dir + '_eval.txt', 'a') as log:
            log.write(message+'\n')
            
    message='total %d:\n'%count
    for k,v in eval_results.items():
        message += 'm_%s: %.5f\t' % (k, v/count)
    print(message)
    with open(save_dir + '_eval.txt', 'a') as log:
        log.write(message+'\n')