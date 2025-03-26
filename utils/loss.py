import torch

def compute_iou(grid_row,grid_col,xywh_a,xywh_b, IMG_SIZE, S):
    grid_size=IMG_SIZE//S
    
    # yolo coordinates
    xcenter_a,ycenter_a,w_a,h_a=xywh_a
    xcenter_b,ycenter_b,w_b,h_b=xywh_b
    
    # normal coordinates
    xcenter_a,ycenter_a=(grid_col+xcenter_a)*grid_size,(grid_row+ycenter_a)*grid_size
    w_a,h_a=w_a*IMG_SIZE,h_a*IMG_SIZE
    xcenter_b,ycenter_b=(grid_col+xcenter_b)*grid_size,(grid_row+ycenter_b)*grid_size
    w_b,h_b=w_b*IMG_SIZE,h_b*IMG_SIZE
    
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

def compute_loss(batch_x,batch_y,batch_output,LAMBDA_NOOBJ,LAMBDA_COORD,S,IMG_SIZE):
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
                
    return loss