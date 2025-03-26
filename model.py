from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights

class Yolov1(nn.Module):
    def __init__(self, S, C):
        super().__init__()
        # 参数初始化
        self.S = S
        self.C = C

        # 加载 ResNet-18 并去掉全连接层
        _resnet18 = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(_resnet18.children())[:-2])  # 去掉最后两层
        for param in self.backbone.parameters():
            param.requires_grad=False
            
        self.head=nn.Sequential(
            nn.Conv2d(in_channels=2048,out_channels=1024,kernel_size=3,padding=1), # (batch,1024,14,14)
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=1,padding=1), # (batch,1024,7,7)
            nn.LeakyReLU(0.1),

            nn.Flatten(),
            nn.Linear(in_features=S*S*1024,out_features=4096),
            nn.Dropout(),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=4096,out_features=S*S*(10+C)),
            nn.Sigmoid()
        )
    
    def forward(self,x): # x:(batch,3,448,448)
        y=self.backbone(x) # y:(batch,512,14,14)
        y=self.head(y) # y:(batch,S*S*(10+C))
        return y.view(-1,self.S,self.S,10+self.C)
        # y.view(-1,self.S,self.S,10+self.C).squeeze(0)[:,:,9].view(-1).max(),y.view(-1,self.S,self.S,10+self.C).squeeze(0)[:,:,4].view(-1).max()