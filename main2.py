from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import cPickle
import os.path
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.preprocessing import normalize

import numpy as np
import cPickle
from sklearn import metrics

IMG_SIZE = 224
SELECTION = 1000

base_model = VGG19(weights='imagenet')
model = Model(
    inputs=base_model.input, outputs=base_model.get_layer('fc2').output)


def get_feature_1_image(image_name):
    img_path = './' + image_name + '.jpg'
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    cPickle.dump(features, open("./test_" + image_name + ".pickle", "wb"))
    print("done for " + image_name)

    pickle_in = open("./test_" + image_name + ".pickle", "rb")
    example_dict = cPickle.load(pickle_in)
    return example_dict


def load_all_feature(select_num):
    print("Load feature")
    all_feature = cPickle.load(open("./allFeature.pickle", "rb"))
    all_label = load_all_labels()
    all_feature_select = SelectKBest(
        chi2, k=select_num).fit(all_feature, all_label).transform(all_feature)

    all_feature_select_norm = normalize(all_feature_select, norm='l2')
    return all_feature_select_norm


def load_all_labels():
    print("loading label")
    all_label = []
    with open("labels.txt", "r") as all_Label_file:
        for i in all_Label_file:
            all_label.append(i.rstrip())
    return all_label


def get_list_file_of_folder(folder_name):
    files = os.listdir(folder_name)
    for name in files:
        print name


def knn(select_num):
    all_feature_data = load_all_feature(select_num)
    all_label = load_all_labels()
    print("start KNN")
    knn = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
    knn.fit(all_feature_data, all_label)

    with open("select" + str(select_num) + ".txt", "wb") as analys_file:
        for test_case in all_feature_data:
            distances, indices = knn.kneighbors([test_case])
            # indices la mot arr dang [[111,123,453,123,1233]]
            # trong do 111 va 123 la index cua tung cai anh mot trong list all feature
            # all_label[indices[0][1]] la lay ra label theo index cua anh do
            analys_file.write(all_label[indices[0][1]] + "\n")


def compare(select_num):
    all_label = []
    with open("labels.txt", "r") as all_Label_file:
        for i in all_Label_file:
            all_label.append(i.rstrip())
    real_result = []
    with open("select" + str(select_num) + ".txt", "r") as real_result_file:
        for i in real_result_file:
            real_result.append(i.rstrip())
    count = 0
    for i in range(8189):
        if all_label[i] == real_result[i]:
            count = count + 1

    print select_num
    print float(count) / 8189


SELECTION = [1500, 2000, 2500, 3000, 3500, 4000]
for target_list in SELECTION:
    knn(target_list)
    compare(target_list)
