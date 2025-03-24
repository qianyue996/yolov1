

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