import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import os

def foreground_detect(img_path):
        # ham nay dung de tao ra mot mask cho anh
        img = cv2.imread(img_path)
        # lay ra chieu dai chieu rong cua anh
        img_height = img.shape[0]
        img_width = img.shape[1]
        # dinh nghia cac chieu cua hình chu nhat
        start_width = int( img_width/10)
        start_height = int(img_height/10)
        rect_width = int(img_width*0.8)
        rect_height = int(img_height*0.8)

        mask = np.zeros(img.shape[:2],np.uint8)

        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)

        rect = (start_width,start_height,rect_width,rect_height)
        # cat phong nen
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        # tra ve mat na mau sac va cac chieu cua anh
        return mask2,img_height,img_width

def find_4_angle(mask2,img_height,img_width):
        # ham nay dung de lay ra 4 canh left,right,top,bottom cua mask
        result_left = []
        result_top = []
        result_right = []

        # lay ra top bottom left
        for i in range(0,img_height,1):
                for j in range(0,img_width,1):
                        if mask2[i][j] != 0:
                                result_top.append(i)
                                result_left.append(j)
                                break
        # lay ra right
        for i in range(0,img_height,1):
                for j in range(img_width-1,-1,-1):
                        if mask2[i][j] != 0:
                                result_right.append(j)
                                break
        # tra ve cac chieu cua anh can crop
        return min(result_left),min(result_top),max(result_right),max(result_top)

def show_img(img_path,find_4_angle):
        img = Image.open(img_path)
        cropped_img = img.crop(find_4_angle)
        cropped_img.save("./crop/"+img_path)

def main(img_path):
        print(img_path)
        mask2,img_height,img_width =  foreground_detect(img_path)
        area = find_4_angle(mask2,img_height,img_width)
        show_img(img_path,area)

main("./2/e5f05b0e-009d-4a96-bbec-9b4c2e6772f6.jpg")

# list_img = os.listdir('./3')
# for i in list_img:
#         main("3/"+i)
