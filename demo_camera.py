import cv2 as cv
import torch
import numpy as np
from torchvision.transforms import ToTensor

from model import Yolov1
from utils.draw import draw_box

device='cpu'
IMG_SIZE=224
S=7
C=1

save_conf=[]

model=Yolov1(S,C).to(device)
model.load_state_dict(torch.load('checkpoint.pth',map_location=device)['model'])
model.eval()

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
                draw_box(frame,i,j,output[i,j,:5],IMG_SIZE,S)
            elif output[i,j,9]>0.5:
                draw_box(frame,i,j,output[i,j,5:10],IMG_SIZE,S)

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