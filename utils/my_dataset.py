import cv2 as cv
from torch.utils.data import Dataset
import torch
from torchvision.transforms import ToTensor
import json
import os

class YoloDataset(Dataset):
    def __init__(self, IMG_SIZE, S, C, image_set='train'):
        super().__init__()
        img_path='data/my_dataset/img'
        xml_path='data/my_dataset/xml'

        self.datasets=[]

        cv_imgs_path=[os.path.join(img_path,i) for i in os.listdir(img_path)]
        bbox_imgs_path=[os.path.join(xml_path, i) for i in os.listdir(xml_path)]
        if len(cv_imgs_path)==len(bbox_imgs_path):
            for i in range(len(cv_imgs_path)):
                cv_img=cv.imread(cv_imgs_path[i])
                with open(bbox_imgs_path[i],'r',encoding='utf-8')as f:
                    label=json.load(f)
                self.datasets.append((cv_img,label))
        else:
            print("数据集长度不一，图像数量不等于标签数量")

        self.IMG_SIZE=IMG_SIZE
        self.S = S
        self.C = C

    def __getitem__(self,index):
        img,label=self.datasets[index]

        x_scaled=self.IMG_SIZE/img.shape[1]
        y_scaled=self.IMG_SIZE/img.shape[0]
        grid_size=self.IMG_SIZE//self.S

        img=cv.resize(img,(self.IMG_SIZE,self.IMG_SIZE))
        x=ToTensor()(img)
        
        # 缩放到448x448时的位置像素
        y=torch.zeros(self.S,self.S,10+self.C)
        label=label['shapes'][0]['points']
        xmin,ymin=int(label[0][0]*x_scaled),int(label[0][1]*y_scaled)
        xmax,ymax=int(label[1][0]*x_scaled),int(label[1][1]*y_scaled)
        xcenter,ycenter=(xmin+xmax)/2,(ymin+ymax)/2
        width,height=xmax-xmin,ymax-ymin
        grid_row,grid_col=int(ycenter//grid_size),int(xcenter//grid_size)

        # yolo 格式数据
        xcenter,ycenter=xcenter%grid_size/grid_size,ycenter%grid_size/grid_size
        width,height=width/self.IMG_SIZE,height/self.IMG_SIZE

        # targets
        y[grid_row,grid_col,0:5]=y[grid_row,grid_col,5:10]=torch.tensor([xcenter,ycenter,width,height,1])   # x,y,w,h,c
        # y[grid_row,grid_col,10:]=torch.zeros(20)
        # y[grid_row,grid_col,10+classid]=1
        y[grid_row,grid_col,10]=1
        return x,y # ((3,448,448),(7,7,30))
    
    def __len__(self):
        return len(self.datasets)
    
    def get_item(self,index):
        img,_=self.datasets[index]
        return img
