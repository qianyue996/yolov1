import cv2 as cv

def draw(img, label):
    # label -> (xmin,ymin,xmax,ymax)
    cv.rectangle(img,(label[0],label[3]))