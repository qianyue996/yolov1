import cv2 as cv

def draw_box(img,row,col,output,IMG_SIZE,S):
    grid_size=IMG_SIZE/S

    # 坐标还原（仅在3x448x448中）
    x,y,w,h,confident=output
    x,y=(x+row)*grid_size,(y+col)*grid_size
    w,h=w*IMG_SIZE,h*IMG_SIZE
    xmin,ymin,xmax,ymax=x-w/2,y-h/2,x+w/2,y+h/2

    # 缩放到原图坐标
    x_scale=img.shape[1]/IMG_SIZE
    y_scale=img.shape[0]/IMG_SIZE
    xmin,ymin,xmax,ymax=xmin*x_scale,ymin*y_scale,xmax*x_scale,ymax*y_scale

    # 画框rectangle
    cv.rectangle(img,(int(xmin),int(ymin)),(int(xmax),int(ymax)),color=(0,0,255))
    # cv.addText(img,)