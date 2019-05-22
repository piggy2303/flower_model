from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import cPickle
import os.path
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


IMG_SIZE = 224
AVA_FOLDER = './demoFolder'

base_model = VGG19(weights='imagenet')

model = Model(
    inputs=base_model.input, outputs=base_model.get_layer('fc2').output)


def get_feature_1_image(image_name):
    img_path = './' + image_name + '.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    cPickle.dump(features, open("./test" + image_name + ".pickle", "wb"))
    print("done for " + image_name)


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


# demo get feature
def get_feature_from_list_of_file(name, type_image):
    print(name, type_image)
    x_ids = os.listdir('./images/'+type_image+'/'+name)
    x_imgs = load_images_from_folder(
        x_ids,
        './images/'+type_image+'/'+name,
        img_size=224,
    )
    x_imgs = preprocess_input(x_imgs)
    features = model.predict(x_imgs)
    cPickle.dump(features, open("./feature/"+name+".pickle", "wb"))


# get_feature_from_list_of_file("flower_validation", "flower")
# get_feature_from_list_of_file("nonflower_test", "nonflower")
# get_feature_from_list_of_file("nonflower_validation", "nonflower")
get_feature_from_list_of_file("nonflower_train", "nonflower")


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
