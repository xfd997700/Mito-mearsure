# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:20:53 2023

@author: Fanding Xu
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import os
from fil_finder import FilFinder2D

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def fig_control(fig):
    ax = plt.Axes(fig,[0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
def th_cut(lens, th1, th2):
    idx = np.logical_and(lens > th1, lens < th2)
    lens = lens[idx]
    return lens    

def result(lens, unit):
    if lens.size != 0:
        avg = lens.mean()
        std = lens.std()
        print("数量 = {:d}".format(lens.size))
        print(("平均长度 = {:.4f}；标准差 = {:.4f}".format(avg, std))+"  单位"+unit)
    else:
        print("无数据，请检查图像或重设阈值")
        
def draw_hist(lens, path):
    fig = plt.figure('hist')
    plt.hist(lens, bins=args.bins)
    plt.xlabel(unit)
    plt.ylabel('amount')
    plt.title('Histogram of the mearsurment')
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mito-mearsure")
    parser.add_argument('--img_path', type=str, default='demo/1.jpg',
                        help='Path of the image'+
                        ' 图像路径')
    parser.add_argument('--out_path', type=str, default='output/',
                        help='Path of the image'+
                        ' 结果输出路径')
    parser.add_argument('--idp', action="store_true",
                        help='Make independent dir for each run'+
                        ' 为每次运行建立独立文件夹')
    parser.add_argument('--bin_th', type=int, default=127,
                        help='Image binaryzation threshold, default as 127'+
                        ' 图像二值化阈值，默认127')
    parser.add_argument('--cm_pos', nargs='+', type=float, default=None,
                        help='The position of coner marker, default as None'+
                        '\n图像角标位置，默认为None')
    parser.add_argument('--scaler', type=int, default=None,
                        help='The scaler of coner marker, default as None'+
                        ' 图像角标尺度数字，默认为None')
    parser.add_argument('--unit', type=str, default=None,
                        help='The scaler of coner marker, default as None'+
                        ' 图像角标尺度单位，默认为None')
    parser.add_argument('--ref', type=float, default=None,
                        help='Reference that converting pixels to unit (1 pixels = ref unit), default as None'+
                        ' 像素到物理单位的换算标准(1 像素 = ref 单位)，默认为None')
    
    parser.add_argument('--th_mode', type=str, default=None,
                        help='The mode of dropping out noise skeletons (pixel: thresholding by pixel amount; unit: thresholding by unit; manual: manually adjust)'+
                        ' 去除干扰骨架的模式，默认为None (pixel: 按像素阈值去除; unit: 按单位阈值去除; manual:手动反复调整)')
    parser.add_argument('--th_l', type=float, default=None,
                        help='Low-cut threshold to dropout skeletons < th_b, default as None'+
                        ' 去除小于低切阈值th_b的骨架，默认为None')
    parser.add_argument('--th_h', type=float, default=None,
                        help='High-cut threshold to dropout skeletons > th_b, default as None'+
                        ' 去除大于高切阈值th_b的骨架，默认为None')
    parser.add_argument('--bins', type=int, default=30,
                        help='Bins of histogram, default as 30'+
                        ' 分布图条数，默认为30')
    args = parser.parse_args()
    
    out_path = args.out_path
        
    assert args.out_path[-1] == '/', "out path must ends with /"
    
    if args.idp:
        run_time = time.strftime("RUN-%Y%m%d-%H%M%S", time.localtime())
        out_path += (run_time + '/')
        mkdir(out_path)
    
    img = cv2.imread(args.img_path, flags=1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(gray, args.bin_th, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    rocm_pos = args.cm_pos

    ref = None
    scaler = args.scaler
    unit = args.unit
    
    if args.ref is not None:
        ref = args.ref
        if unit is None:
            unit = input("请输入物理单位\nPlease input the unit: ")
        print("1 pixel = {:.4f} ".format(ref) + unit)
    elif rocm_pos is not None:
        rocm = binary[round(binary.shape[0]*rocm_pos[0]):round(binary.shape[0]*rocm_pos[1]),
                      round(binary.shape[1]*rocm_pos[0]):round(binary.shape[1]*rocm_pos[1])].copy()

        binary[round(binary.shape[0]*rocm_pos[0]):round(binary.shape[0]*rocm_pos[1]),
                round(binary.shape[1]*rocm_pos[0]):round(binary.shape[1]*rocm_pos[1])] = 0

        cont_rocm, _ = cv2.findContours(rocm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if scaler is None:
            scaler = int(input("请输入角标尺度\nPlease input the scaler: "))
        if unit is None:
            unit = input("请输入角标单位\nPlease input the unit: ")
        cont_rocm = list(cont_rocm)
        cont_rocm.sort(key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
        markers = np.array([cv2.boundingRect(m)[2] for m in cont_rocm])
        marker_idx = markers.argmax()
        marker = markers[marker_idx]
        rocm_edge = cv2.rectangle(cv2.cvtColor(rocm, cv2.COLOR_GRAY2BGR), cv2.boundingRect(cont_rocm[marker_idx]), (0,255,0), 2)
        # plt.imshow(rocm_edge)

        ref = scaler / marker
        print(("Reference: {:d} "+unit+" = {:d} pixels").format(scaler, marker))
        print("1 pixel = {:.4f} ".format(ref) + unit)
        cv2.imwrite(out_path + "region of marker.png", rocm_edge)
    else:
        ref = 1
        unit = 'pixels'
        
    cv2.imwrite(out_path + "binary.png", binary)



    contours,hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    edge = cv2.drawContours(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), contours, -1, (255, 0, 0), 4)
    
    dst = binary.copy()
    
    # Step 1: Create an empty skeleton
    size = np.size(dst)
    skel = np.zeros(dst.shape, np.uint8)
    
    # Get a Cross Shaped Kernel
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    
    print("正在提取骨架... Plotting skeleton...")
    print("(请忽略警告 Ignore the warning)")
    # Repeat steps 2-4
    while True:
        op = cv2.morphologyEx(dst, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(dst, op)
        eroded = cv2.erode(dst, element)
        skel = cv2.bitwise_or(skel,temp)
        dst = eroded.copy()
        if cv2.countNonZero(dst)==0:
            break
    
    
    fil = FilFinder2D(binary, mask=skel)
    fil.preprocess_image(flatten_percent=85)
    fil.create_mask(border_masking=True, verbose=False,
    use_existing_mask=True)
    fil.medskel(verbose=False)
    filsk = fil.skeleton
    print("Done!")
    #%%
    size = (binary.shape[1]/100, binary.shape[0]/100)
    
    print("正在绘制骨架... Plotting skeleton...")
    fig = plt.figure('main', figsize=size)
    fig_control(fig)
    plt.imshow(edge)
    plt.contour(filsk, colors='g', linewidths=0.2)
    fig.savefig(out_path + 'skeleton.png')
    plt.close(fig)
    print("Done!")
    print("骨架图已保存 Skeleton saved：" + out_path + 'skeleton.png')

    sk = filsk.astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(sk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    lens_pix = np.array([cv2.arcLength(m, True) for m in contours])
    lens = lens_pix * ref
    
    
        
    #%%
    draw_hist(lens, out_path + 'histogram.png')
    result(lens, unit)
    
    th_mode = args.th_mode
    if th_mode is not None:
        assert th_mode in ['unit', 'pixel', 'manual']
        th1 = args.th_l
        th2 = args.th_h
        
        print("最小长度 Minimum length: {:.2f} pixels; 最大长度 maximum length: {:.2f} pixels".format(lens_pix.min(), lens_pix.max()))
        if unit != 'pixels':
            print(("最小长度 Minimum length: {:.2f} "+unit+"; 最大长度 maximum length: {:.2f} "+unit).format(lens_pix.min()*ref, lens_pix.max()*ref))
        print("去扰模式 cut mode: "+th_mode)   
        
        if th_mode != 'manual':
            if th1 is None:
                th1 = float(input("输入低切阈值:\nInput low-cut threshold: "))
            if th2 is None:
                th2 = float(input("输入高切阈值:\nInput low-cut threshold: "))
            if th_mode == 'pixel':
                lens_pix = th_cut(lens_pix, th1, th2)
                lens = lens_pix * ref
            else:
                lens = lens_pix * ref
                lens = th_cut(lens, th1, th2)


        else:
            lens_keep = lens_pix.copy()
            while(True):
                print("进入手动调整模式，输入 over 结束调整")
                lens_pix = lens_keep
                
                
                mode = input("输入切割依据\n Input cut mode (0=pixel, 1=unit): ")
                if mode == 'over':
                    print("退出手动模式")
                    break
                else:
                    mode = int(mode)
                if mode not in [0, 1]:
                    mode = int(input("重新输入，必须为0或1: "))

                th1 = input("输入低切阈值:\nInput low-cut threshold: ")
                if th1 == 'over':
                    print("退出手动模式")
                    break
                else:
                    th1 = float(th1)
                
                th2 = input("输入高切阈值:\nInput low-cut threshold: ")
                if th2 == 'over':
                    print("退出手动模式")
                    break
                else:
                    th2 = float(th2)
                    
                if mode == 0:
                    lens_pix = th_cut(lens_pix, th1, th2)
                    lens = lens_pix * ref
                    draw_hist(lens_pix, out_path + 'histogram_manual.png')
                else:
                    lens = lens_pix * ref
                    lens = th_cut(lens, th1, th2)
                    draw_hist(lens, out_path + 'histogram_manual.png')
                result(lens, unit)
                print("直方图已存储，请参考"+out_path + 'histogram_manual.png')
                
    draw_hist(lens, out_path + 'histogram.png')
    result(lens, unit)
    with open(out_path+'result.txt', 'w') as file_object:
        result = "数量 = {:d}\n".format(lens.size)+("平均长度 = {:.4f}；标准差 = {:.4f}".format(lens.mean(), lens.std()))+"  单位"+unit
        file_object.write(result)

















































































































































































