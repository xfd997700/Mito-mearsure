# 线粒体长度统计程序 Mito-mearsure V1.0 使用说明

## 环境要求：
Python-3.9.7

cv2-4.7.0

matplotlib-3.4.3

numpy-1.20.3

argparse-1.1

fil_finder-1.7.2

推荐直接使用附带的虚拟环境venv:
通过下面指令进行激活
>     venv\Scripts\activate

## 使用方法：
直接调用mearsure.py进行图像处理与统计
>     python mearsure.py

>可通过 -h 查看并获得帮助
>     python mearsure.py -h

基本初始变量包括：
>--img_path 要处理的图像路径，如
>     python mearsure.py --img_path demo/1.jpg

>--out_path 处理结果输出路径，如
>     python mearsure.py --out_path output/

>--idp 调用则为每次运行结果创立独立文件夹
>--bin_th 图像二值化的阈值，默认为127

单位标定相关变量包括：
>--cm_pos 通过图像上的角标自动标定，输入角标的位置，如
>     python mearsure.py --cm_pos 0.95 1 0.95 1

>其中四个数字0.95, 1, 0.95, 1代表了角标区域的范围，即纵坐标从图像的95%处到100%处，横坐标从图像的95%处到100%处。

>--scaler 图像角标的数字

>--unit 要换算的目标物理单位

>--ref 直接设定像素到物理单位的换算标准，如下表示1像素=0.09微米

>     python mearsure.py --ref 0.09 --unit um

数据统计相关变量包括：
>--th_mode 设置去除干扰噪声的模式，默认为None即不进行去扰，可设置为‘pixel’，‘unit’和‘manual’三种模式，‘pixel’和‘unit’需固定阈值，‘pixel’为根据像素数量直接进行去扰；‘unit’为根据设置的物理单位直接进行去扰；‘manual’为手动进行反复调整，每次调整都会给出直方图供参考。示例：

>     python mearsure.py --th_mode manual

>--th_l 设置低切阈值，去除小于低切阈值th_b的骨架

>--th_h 设置高切阈值，去除小于高切阈值th_b的骨架

>上述两个变量如未定义，程序也会在运行后要求手动输入

>--bins 设置统计直方图的bar数量，默认为30
