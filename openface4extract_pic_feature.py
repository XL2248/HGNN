# -*- coding:utf8 -*-
import cv2
import os,code
import shutil


# input_file = '\"E:\\EMNLP2020\\MELD.Raw\\dev_splits\\dia0_utt0.mp4"'
# output_dir = '\"E:\\EMNLP2020\\MELD.Raw\\dev_splits\\'

# code.interact(local=locals())
current_file_type = 'train4_' # train test dev
#视频文件名字
source_path = 'E:/EMNLP2020/MELD.Raw/' + current_file_type +'splits/'
target_path = 'E:/EMNLP2020/MELD.Raw/' + current_file_type +'features/'
#视频帧率12
fps = 12
#保存图片的帧率间隔
count = 1
# print('filepath', filename)
j = 0
# 获取当前文件夹下所有文件名称
files= os.listdir(source_path)
for filename in files: #遍历文件夹
     if not os.path.isdir(filename): #判断是否是文件夹，不是文件夹才打开
          # f = open(source_path+"/"+file); #打开文件
        j += 1
        #保存feature的路径
        savedpath = target_path + filename.split('.')[0] + '/'
        isExists = os.path.exists(savedpath)
        if not isExists:
            os.makedirs(savedpath)
            print('path of %s is build'%(savedpath))
        else:
            shutil.rmtree(savedpath)
            os.makedirs(savedpath)
            print('path of %s already exist and rebuild'%(savedpath))\

        input_file = source_path + filename
        output_dir = target_path + filename.split('.')[0] + '/'
        
        command = 'C:/Users/yunlonliang/Downloads/OpenFace_2.2.0_win_x64/FeatureExtraction.exe -f ' + input_file + ' -out_dir ' + output_dir
        print(command)
        os.system(command)