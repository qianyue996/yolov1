from torchvision.datasets import VOCDetection
from torch.utils.data import Dataset
import torch
from torchvision.transforms import ToTensor

class YoloVOCDataset(Dataset):
    def __init__(self, IMG_SIZE, S, C, image_set='train'):
        super().__init__()
        self.voc_ds=VOCDetection(root='./data',year='2012',image_set=image_set,download=False)
        
        classdict=set()
        for _,label in self.voc_ds:
            for obj in label['annotation']['object']:
                classdict.add(obj['name'])
        names=sorted(list(classdict))
        self.id2name={i:c for i,c in enumerate(names)}
        self.name2id={c:i for i,c in self.id2name.items()}

        self.IMG_SIZE = IMG_SIZE
        self.S = S
        self.C = C
    
    def __getitem__(self,index):
        img,label=self.voc_ds[index]
    
        x_scale=self.IMG_SIZE/img.width
        y_scale=self.IMG_SIZE/img.height
        grid_size=self.IMG_SIZE//self.S
        
        scaled_img=img.resize((self.IMG_SIZE,self.IMG_SIZE))
        x=ToTensor()(scaled_img)
        y=torch.zeros(self.S,self.S,10+self.C)
        
        for obj in label['annotation']['object']:
            box=obj['bndbox']
            classid=self.name2id[obj['name']]
            
            # normal coordinates
            xmin,ymin,xmax,ymax=int(box['xmin'])*x_scale,int(box['ymin'])*y_scale,int(box['xmax'])*x_scale,int(box['ymax'])*y_scale
            xcenter,ycenter=(xmin+xmax)/2,(ymin+ymax)/2
            width,height=xmax-xmin,ymax-ymin
            grid_i,grid_j=int(ycenter//grid_size),int(xcenter//grid_size)
            
            # yolo coordinates
            xcenter,ycenter=xcenter%grid_size/grid_size,ycenter%grid_size/grid_size
            width,height=width/self.IMG_SIZE,height/self.IMG_SIZE
            
            # targets
            y[grid_i,grid_j,0:5]=y[grid_i,grid_j,5:10]=torch.tensor([xcenter,ycenter,width,height,1])   # x,y,w,h,c
            y[grid_i,grid_j,10:]=torch.zeros(20)
            y[grid_i,grid_j,10+classid]=1
        return x,y # ((3,448,448),(7,7,30))
    
    def __len__(self):
        return len(self.voc_ds)
