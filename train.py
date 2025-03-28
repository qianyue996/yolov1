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

class Trainer():
    def __init__(self):
        super().__init__()
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.IMG_SIZE=224
        self.S=7
        self.C=1
        self.LAMBDA_COORD=5
        self.LAMBDA_NOOBJ=0.5
        self.lr=3e-5
        self.epochs=300

        self.grid_size=self.IMG_SIZE//self.S

        self.losses=[]

        self.checkpoint=None

    def setup(self):
        self.model=Yolov1(self.S, self.C).to(self.device)
        self.optimizer=optim.Adam([param for param in self.model.parameters() if param.requires_grad],lr=self.lr)

        # 尝试从上次训练结束点开始
        # try:
        #     checkpoint=torch.load('checkpoint.pth')
        # except Exception as e:
        #     pass
        if self.checkpoint:
            self.model.load_state_dict(self.checkpoint['model'])
        if self.checkpoint:    
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])

        # 加载数据集
        ds=YoloDataset(self.IMG_SIZE, self.S, self.C)
        # ds=YoloVOCDataset(IMG_SIZE, S, C)

        self.dataloader=DataLoader(ds,batch_size=4,shuffle=True)

        # tensorboard
        self.writer=SummaryWriter(f'runs/{time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())}')

        for param in self.model.backbone.parameters():
            param.requires_grad = False

        self.model.train()

    def train(self):
        batch_avg_loss=0
        for epoch in range(self.epochs):
            with tqdm(self.dataloader, disable=False) as bar:
                for batch_x,batch_y in bar:
                    batch_x,batch_y=batch_x.to(self.device),batch_y.to(self.device)
                    batch_output=self.model(batch_x)
                    loss=self.compute_loss(batch_x,batch_y,batch_output)
                    loss=loss/len(batch_x)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    batch_avg_loss+=loss.item()
                    bar.set_postfix({'epoch':epoch,
                                     'loss:':loss.item()})
        
            batch_avg_loss=batch_avg_loss/len(self.dataloader)
            tqdm.write(f"本epoch平均损失为: {batch_avg_loss}")
            self.writer.add_scalar('epoch_loss',batch_avg_loss,epoch)
            self.losses.append(batch_avg_loss,)

    def compute_loss(self,batch_x,batch_y,batch_output):
        loss=torch.tensor(0)
        for i in range(len(batch_x)):
            x=batch_x[i]
            y=batch_y[i]
            output=batch_output[i]
            # foreach grid
            for row in range(self.S):    
                for col in range(self.S):
                    pred_grid=output[row,col]
                    target_grid=y[row,col]
                    if not target_grid[4]>0: # no object in this grid
                        loss_c_noobj=(pred_grid[4])**2+(pred_grid[9])**2 # no object in grid,so target c is 0
                        loss=loss+self.LAMBDA_NOOBJ*loss_c_noobj
                        continue
                    # IOU
                    iou_bbox1=self.compute_iou(row,col,pred_grid[:4],target_grid[:4])
                    iou_bbox2=self.compute_iou(row,col,pred_grid[5:9],target_grid[:4])
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
                    loss=loss+loss_xywh*self.LAMBDA_COORD+loss_c_obj+loss_c_noobj*self.LAMBDA_NOOBJ+loss_class
        return loss

    def compute_iou(self,grid_row,grid_col,xywh_a,xywh_b):
        # yolo coordinates
        xcenter_a,ycenter_a,w_a,h_a=xywh_a
        xcenter_b,ycenter_b,w_b,h_b=xywh_b
        
        # normal coordinates
        xcenter_a,ycenter_a=(grid_row+xcenter_a)*self.grid_size,(grid_col+ycenter_a)*self.grid_size # grid_col and grid_row交换位置
        w_a,h_a=w_a*self.IMG_SIZE,h_a*self.IMG_SIZE
        xcenter_b,ycenter_b=(grid_row+xcenter_b)*self.grid_size,(grid_col+ycenter_b)*self.grid_size
        w_b,h_b=w_b*self.IMG_SIZE,h_b*self.IMG_SIZE
        
        # border
        xmin_a,xmax_a,ymin_a,ymax_a=xcenter_a-w_a/2,xcenter_a+w_a/2,ycenter_a-h_a/2,ycenter_a+h_a/2
        xmin_b,xmax_b,ymin_b,ymax_b=xcenter_b-w_b/2,xcenter_b+w_b/2,ycenter_b-h_b/2,ycenter_b+h_b/2
        
        # IOU
        inter_xmin=max(xmin_a,xmin_b)
        inter_xmax=min(xmax_a,xmax_b)
        inter_ymin=max(ymin_a,ymin_b)
        inter_ymax=min(ymax_a,ymax_b)
        if inter_xmax<inter_xmin or inter_ymax<inter_ymin:
            return 0

        inter_area=(inter_xmax-inter_xmin)*(inter_ymax-inter_ymin) # 交集
        union_area=w_a*h_a+w_b*h_b-inter_area # 并集

        return inter_area/union_area # IOU

    def save_best_model(self):
        if len(self.losses)==1 or self.losses[-1]<self.losses[-2]: # 保存更优的model
            torch.save({'model':self.model.state_dict(),
                        'optimizer':self.optimizer.state_dict()},'.checkpoint.pth')
            os.replace('.checkpoint.pth','checkpoint.pth')

        EARLY_STOP_PATIENCE=5   # 早停忍耐度
        if len(self.losses)>=EARLY_STOP_PATIENCE:
            early_stop=True
            for i in range(1,EARLY_STOP_PATIENCE):
                if self.losses[-i]<self.losses[-i-1]:
                    early_stop=False
                    break
                if early_stop:
                    print(f'early stop, final loss={self.losses[-1]}')
                    break
    

if __name__ == '__main__':
    trainer=Trainer()
    trainer.setup()
    trainer.train()