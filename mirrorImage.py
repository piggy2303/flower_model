from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import cPickle


def filp_image(img_path, img_name):
    datagen = ImageDataGenerator(
        vertical_flip=True,
        horizontal_flip=True,
    )
    img = load_img(img_path)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1, ) + x.shape)
    i = 0

    for batch in datagen.flow(
            x,
            batch_size=32,
            shuffle=False,
            save_to_dir='./RF_data/valid_mirror',
            save_prefix=img_name,
            save_format='jpg'):
        i += 1
        if i > 2:
            break  # otherwise the generator would loop indefinitely


def get_image_from_file(file_path):
    files = os.listdir(file_path)
    print(len(files))
    for name in files:
        name_image = name.split(".")
        filp_image('./RF_data/valid/' + name, name_image[0])


get_image_from_file("./RF_data/valid")
