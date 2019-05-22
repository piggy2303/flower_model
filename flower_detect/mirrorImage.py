# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os


# def filp_image(img_path, image_name):
#     datagen = ImageDataGenerator(
#         vertical_flip=True,
#         horizontal_flip=True,
#     )
#     img = load_img(img_path)  # this is a PIL image
#     x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
#     x = x.reshape((1, ) + x.shape)
#     i = 0

#     for batch in datagen.flow(
#             x,
#             batch_size=32,
#             shuffle=False,
#             save_to_dir='mirror_images',
#             save_prefix="mirror_" + image_name,
#             save_format='jpg'):
#         i += 1
#         if i > 3:
#             break  # otherwise the generator would loop indefinitely


# def test():
#     files = os.listdir('./images/flower')
#     for name in files:
#         image_name = name.split(".")[0]
#         filp_image('./images/flower/' + name, image_name)


def move_file(param):
    file_in_folder_1 = os.listdir('./images/'+param)
    for name1 in file_in_folder_1:
        os.rename("./images/"+param+"/"+name1,
                  "./images/nonflower/"+param+"_"+name1)


# move_file("machine")

def rename(param):

    with open("./slipt_data_list/"+param+".txt") as target:
        #     files_list = os.listdir('./images/flower')
        for line in target:
            line = str(line.rstrip())
            print('./images/flower/image_' + line + '.jpg',
                  './images/flower/'+param+'/image_'+line+'.jpg')
            os.rename('./images/flower/image_'+line+'.jpg',
                      './images/flower/'+param+'/image_'+line+'.jpg')


rename("flower_validation")
