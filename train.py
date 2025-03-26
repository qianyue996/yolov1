from torch.utils.data.dataloader import DataLoader
from torch import optim
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm

from utils.my_dataset import YoloDataset
from datasets import YoloVOCDataset
from model import Yolov1
from utils.loss import *

def main():
    # 定义使用设备是gpu or cpu
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print('\n')
    print('='*20)
    print(f'当前训练选择的是：{device}')
    print('='*20,'\n')

    # 输入图像大小
    IMG_SIZE=224  # resnet backbone决定的
    # yolov1分为7x7个格子
    S=7
    # 20个分类
    C=1
    # loss lambda倍率
    LAMBDA_COORD=5
    LAMBDA_NOOBJ=0.5

    checkpoint=None
    # 尝试从上次训练结束点开始
    # try:
    #     checkpoint=torch.load('checkpoint.pth')
    # except Exception as e:
    #     pass

    model=Yolov1(S, C).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        
    optimizer=optim.Adam([param for param in model.parameters() if param.requires_grad],lr=3e-5)
    if checkpoint:    
        optimizer.load_state_dict(checkpoint['optimizer'])
    # 加载数据集
    ds=YoloDataset(IMG_SIZE, S, C)
    dataloader=DataLoader(ds,batch_size=1,shuffle=True)

    # tensorboard
    writer=SummaryWriter(f'runs/{time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())}')

    model.train()
    losses=[]
    for epoch in range(500):
        batch_avg_loss=0
        with tqdm(dataloader) as bar:
            for batch_x,batch_y in bar:
                batch_x,batch_y=batch_x.to(device),batch_y.to(device)
                batch_output=model(batch_x)
                
                loss=compute_loss(batch_x,batch_y,batch_output,LAMBDA_NOOBJ,LAMBDA_COORD,S,IMG_SIZE)
                loss=loss/len(batch_x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_avg_loss+=loss.item()
                bar.set_postfix({'loss:':batch_avg_loss.item()})
        
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
