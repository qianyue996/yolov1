from torch.utils.data.dataloader import DataLoader
from torch import optim
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm

from datasets import YoloVOCDataset
from model import Yolov1
from loss import *

def main():
    # 定义使用设备是gpu or cpu
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print('\n','='*20)
    print(f'当前训练选择的是：{device}')
    print('='*20,'\n')

    # 输入图像大小
    IMG_SIZE=448  # resnet backbone决定的
    # yolov1分为7x7个格子
    S=7
    # 20个分类
    C=20
    # loss lambda倍率
    LAMBDA_COORD=5
    LAMBDA_NOOBJ=0.5

    checkpoint=None
    # 尝试从上次训练结束点开始
    try:
        checkpoint=torch.load('checkpoint.pth')
    except Exception as e:
        pass

    model=Yolov1(S, C).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        
    optimizer=optim.Adam([param for param in model.parameters() if param.requires_grad],lr=3e-5)
    if checkpoint:    
        optimizer.load_state_dict(checkpoint['optimizer'])
    # 加载数据集
    ds=YoloVOCDataset(IMG_SIZE, S, C)
    dataloader=DataLoader(ds,batch_size=1,shuffle=True)

    # tensorboard
    writer=SummaryWriter(f'runs/{time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())}')

    model.train()
    losses=[]
    for epoch in range(500):
        batch_avg_loss=0
        for batch_x,batch_y in tqdm(dataloader):
            batch_x,batch_y=batch_x.to(device),batch_y.to(device)
            batch_output=model(batch_x)
            
            loss=torch.tensor(0)
            for i in range(len(batch_x)):
                x=batch_x[i]
                y=batch_y[i]
                output=batch_output[i]
                # foreach grid 
                for row in range(S):    
                    for col in range(S):
                        pred_grid=output[row,col] 
                        target_grid=y[row,col]
                        if not target_grid[4]>0: # no object in this grid
                            loss_c_noobj=(pred_grid[4])**2+(pred_grid[9])**2 # no object in grid,so target c is 0
                            loss=loss+LAMBDA_NOOBJ*loss_c_noobj
                            continue 
                        # IOU
                        iou_bbox1=compute_iou(row,col,pred_grid[:4],target_grid[:4],IMG_SIZE,S)
                        iou_bbox2=compute_iou(row,col,pred_grid[5:9],target_grid[:4],IMG_SIZE,S)
                        # 取IOU大的预测框的x,y,w,h,c
                        if iou_bbox1>iou_bbox2:
                            xywh=pred_grid[:4]
                            c_obj,c_noobj=pred_grid[4],pred_grid[9]
                            iou_obj,iou_noobj=iou_bbox1,iou_bbox2
                        else:
                            xywh=pred_grid[5:9]
                            c_obj,c_noobj=pred_grid[9],pred_grid[4]
                            iou_obj,iou_noobj=iou_bbox2,iou_bbox1
                        loss_xywh=(xywh[0]-target_grid[0])**2+(xywh[1]-target_grid[1])**2+(torch.sqrt(xywh[2])-torch.sqrt(target_grid[2]))**2+(torch.sqrt(xywh[3])-torch.sqrt(target_grid[3]))**2
                        loss_c_obj=(c_obj-iou_obj)**2
                        loss_c_noobj=c_noobj**2
                        loss_class=((pred_grid[10:]-target_grid[10:])**2).sum()
                        loss=loss+loss_xywh*LAMBDA_COORD+loss_c_obj+loss_c_noobj*LAMBDA_NOOBJ+loss_class
            loss=loss/len(batch_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_avg_loss+=loss.item()
        
        batch_avg_loss=batch_avg_loss/len(dataloader)
        losses.append(batch_avg_loss)
        
        if len(losses)==1 or losses[-1]<losses[-2]: # 保存更优的model
            torch.save({'model':model.state_dict(),
                        'optimizer':optimizer.state_dict()},'.checkpoint.pth')
            os.replace('.checkpoint.pth','checkpoint.pth')
        writer.add_scalar('loss',losses[-1],epoch) # tersorboard
        
        EARLY_STOP_PATIENCE=5   # 早停忍耐度
        if len(losses)>=EARLY_STOP_PATIENCE:
            early_stop=True
            for i in range(1,EARLY_STOP_PATIENCE):
                if losses[-i]<losses[-i-1]:
                    early_stop=False
                    break
            if early_stop:
                print(f'early stop, final loss={losses[-1]}')
                break

if __name__ == '__main__':
    main()