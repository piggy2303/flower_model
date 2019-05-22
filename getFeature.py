from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import cPickle
import os.path
import os

import cv2
from matplotlib import pyplot as plt
from PIL import Image


IMG_SIZE = 224
AVA_FOLDER = './demoFolder'

base_model = VGG19(weights='imagenet')

model = Model(
    inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

def foreground_detect(img_path):
        # ham nay dung de tao ra mot mask cho anh
        img = cv2.imread(img_path)

        img_height = img.shape[0]
        img_width = img.shape[1]

        start_width = int( img_width/10)
        start_height = int(img_height/10)
        rect_width = int(img_width*0.8)
        rect_height = int(img_height*0.8)

        mask = np.zeros(img.shape[:2],np.uint8)

        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)

        rect = (start_width,start_height,rect_width,rect_height)

        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

        return mask2,img_height,img_width

def find_4_angle(mask2,img_height,img_width):
        # ham nay dung de lay ra 4 canh left,right,top,bottom cua mask
        result_left = []
        result_top = []
        result_right = []
        for i in range(0,img_height,1):
                for j in range(0,img_width,1):
                        if mask2[i][j] != 0:
                                result_top.append(i)
                                result_left.append(j)
                                break
        for i in range(0,img_height,1):
                for j in range(img_width-1,-1,-1):
                        if mask2[i][j] != 0:
                                result_right.append(j)
                                break
        return min(result_left),min(result_top),max(result_right),max(result_top)

def show_img(img_path,find_4_angle):
        

def foreground_cut(img_path):
        mask2,img_height,img_width =  foreground_detect(img_path)
        area = find_4_angle(mask2,img_height,img_width)
        img = Image.open(img_path)
        cropped_img = img.crop(area)
        cropped_img.show()
        return cropped_img

def get_feature_1_image(image_name):
    img_path = './' + image_name + '.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    cPickle.dump(features, open("./test" + image_name + ".pickle", "wb"))
    print("done for " + image_name)


# get_feature_1_image("image_06734")
# get_feature_1_image("image_08047")


def test():
    all_feature_1 = cPickle.load(open("testimage_06734.pickle", "rb"))
    all_feature_2 = cPickle.load(open("testimage_08047.pickle", "rb"))

    print(all_feature_1)
    print(all_feature_2)


def load_images_from_folder(imglist, folderName, img_size=IMG_SIZE):
    Xs = []
    for img in imglist:
        img_path = os.path.join(folderName, img)
        im = image.load_img(img_path, target_size=(img_size, img_size))
        im = image.img_to_array(im)
        # im = im.transpose((2, 0, 1))
        Xs.append(im)
    Xs = np.asarray(Xs)
    return Xs


def get_feature_folder(src_folder, des_file_feature):
    x_ids = []

    x_ids = os.listdir(src_folder)

    print(x_ids)
    x_imgs = load_images_from_folder(
        x_ids,
        src_folder,
        img_size=224,
    )
    x_imgs = preprocess_input(x_imgs)
    features = model.predict(x_imgs)
    cPickle.dump(features, open(des_file_feature, "wb"))


# list_folder = os.listdir("./anh_thuc_te")

# for item in list_folder:
#     print("./anh_thuc_te/"+item)
#     print("./anh_thuc_te/"+item+".pickle")
#     get_feature_folder("./anh_thuc_te/"+item,
#                        "./anh_thuc_te/"+item+".pickle")


def train_mirror():
    list_mirror = os.listdir('./RF_data/valid_mirror')
    with open("data_valid_mirror.txt", "wb") as file_mirror:
        with open("data_valid.txt", "r") as file:
            for i in file:
                i = int(i.rstrip())
                if i < 10:
                    i = "0000"+str(i)
                if 10 <= i < 100:
                    i = "000"+str(i)
                if 100 <= i < 1000:
                    i = "00"+str(i)
                if 1000 <= i < 10000:
                    i = "0"+str(i)
                print("finding " + i)
                for j in list_mirror:
                    image_name = j
                    number_image = j.split("_")
                    # print(number_image, i)
                    if number_image[1] == i:
                        print(image_name)
                        file_mirror.write(image_name+"\n")


# train_mirror()


def get_all_feature():
    with open("list_of_plant.txt", "r") as list_of_plant:
        for plant_name in list_of_plant:
            plant_name = plant_name.rstrip()
            folder_name = "./images/" + plant_name
            print("doing in folder" + folder_name)

            list_of_images = []
            with open("./list_image/" + plant_name + ".txt",
                      "r") as list_of_image_in_folder:
                print("plant: " + plant_name)
                for image_name in list_of_image_in_folder:
                    image_name = image_name.rstrip()
                    list_of_images.append(image_name)

            # print list_of_images
            x_ids = list_of_images
            x_imgs = load_images_from_folder(
                x_ids,
                folder_name,
                img_size=224,
            )
            x_imgs = preprocess_input(x_imgs)
            features = model.predict(x_imgs)
            cPickle.dump(features,
                         open("./feature/" + plant_name + ".pickle", "wb"))
            print("done ./feature/" + plant_name + ".pickle")


# get_feature_1_image("image_06757")
# get_feature_from_list_of_file()

