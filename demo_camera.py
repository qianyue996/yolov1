import cv2 as cv
import torch
import numpy as np
from torchvision.transforms import ToTensor

from model import Yolov1

device='cpu'
IMG_SIZE=224
S=7
C=1

save_conf=[]

model=Yolov1(S,C).to(device)
model.load_state_dict(torch.load('checkpoint.pth',map_location=device)['model'])
model.eval()

def draw_box(img,row,col,output):
    grid_size=IMG_SIZE/S

    # 坐标还原（仅在3x448x448中）
    cx,cy=grid_size*(col+output[0]),grid_size*(row+output[1])
    w,h=output[2]*IMG_SIZE,output[3]*IMG_SIZE
    xmin,ymin,xmax,ymax=cx-w/2,cy-h/2,cx+w/2,cy+h/2

    # 缩放到原图坐标
    x_scale=img.shape[1]/IMG_SIZE
    y_scale=img.shape[0]/IMG_SIZE
    cx,cy=cx*x_scale,cy*y_scale
    xmin,ymin,xmax,ymax=xmin*x_scale,ymin*y_scale,xmax*x_scale,ymax*y_scale

    # 画框rectangle
    cv.rectangle(img,(int(xmin),int(ymin)),(int(xmax),int(ymax)),color=(0,0,255))

def process_frame(frame,model):
    global save_conf

    input=ToTensor()(frame)

    output=model(input.unsqueeze(0))[0]
    
    save_conf.append(output[:,:,4].max())
    save_conf.append(output[:,:,9].max())
    if len(save_conf)>10:
        print(max(save_conf))
        save_conf=[]

    for i in range(S):
        for j in range(S):
            if output[i,j,4]>0.5:
                draw_box(frame,i,j,output[i,j,:5])
            if output[i,j,9]>0.5:
                draw_box(frame,i,j,output[i,j,5:10])

if __name__=='__main__':
    cap = cv.VideoCapture(0)

    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("无法获取帧！")
            break
        frame=cv.resize(frame,(IMG_SIZE,IMG_SIZE))
        process_frame(frame,model)
        frame=cv.resize(frame,(540,448))
        cv.imshow('Camera', frame)
        if cv.waitKey(1) == ord('q'):
            break
    # 释放资源
    cap.release()
    cv.destroyAllWindows()