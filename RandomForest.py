from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.decomposition import PCA
import numpy as np
import cPickle
import os


def load_all_feature(path):
    print("Load feature" + path)
    all_feature = cPickle.load(open(path, "rb"))
    all_feature_norm = normalize(all_feature, norm='l2')
    return all_feature_norm


def load_all_labels(path):
    print("loading label")
    all_label = []
    with open(path, "r") as all_Label_file:
        for i in all_Label_file:
            all_label.append(int(i.rstrip()))
    return all_label


def load_test():
    print("Load test")
    all_feature = cPickle.load(open("./test_image_06734.pickle", "rb"))
    all_feature_norm = normalize(all_feature, norm='l2')
    return all_feature_norm


def slipt_data(type):
    with open("data_"+type+".txt") as file_text:
        for i in file_text:
            i = int(i.rstrip())
            if i < 10:
                i = "0000"+str(i)
            if 10 <= i < 100:
                i = "000"+str(i)
            if 100 <= i < 1000:
                i = "00"+str(i)
            if 1000 <= i < 10000:
                i = "0"+str(i)

            os.rename("./RF_data/jpg/image_"+i+".jpg",
                      "./RF_data/"+type+"/image_"+i+".jpg")


def generate_label():
    arr = []
    for i in range(1, 103, 1):
        for j in range(0, 10, 1):
            arr.append(i)
    # print(len(arr))
    return arr


def generate_label_mirror():
    arr = []
    for i in range(1, 103, 1):
        for j in range(0, 30, 1):
            arr.append(i)
    # print(len(arr), arr)
    return arr


path_to_model_pca = "./model_saving/model_pca_1.sav"


def pca_training(X, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    print("save pca")
    cPickle.dump(pca, open(path_to_model_pca, 'wb'))


def pca_transform(X):
    print("use pca " + path_to_model_pca)
    pca = cPickle.load(open(path_to_model_pca, 'rb'))
    return pca.transform(X)


def svc_training(path):
    X_train_init = load_all_feature('./RF_feature/train.pickle')
    X_train_mirror = load_all_feature("./RF_feature/train_mirror.pickle")
    X_val_init = load_all_feature("./RF_feature/valid.pickle")
    X_val_mirror = load_all_feature("./RF_feature/valid_mirror.pickle")

    X_test_01 = load_all_feature("./RF_feature/test_01.pickle")
    X_test_02 = load_all_feature("./RF_feature/test_02.pickle")

    X_train = np.concatenate((
        X_train_init,
        X_train_mirror,
        X_val_init,
        X_val_mirror,
        X_test_01,
        X_test_02
    ), axis=0)

    # pca_training(X=X_train, n_components=256)
    # X_train = pca_transform(X=X_train)

    # print(X_train.shape)
    Y_test_01 = load_all_labels("./RF_data/label_test_1.txt")
    Y_test_02 = load_all_labels("./RF_data/label_test_2.txt")

    Y_train = generate_label() + generate_label_mirror() + generate_label() + \
        generate_label_mirror()+Y_test_01+Y_test_02

    clf = LinearSVC(
        random_state=0,
        tol=1e-5,
        C=2,
    )

    print("training model")
    clf.fit(X_train, Y_train)

    print("save model")
    cPickle.dump(clf, open(path, 'wb'))


def using_model(model_name, pickle_name, image_label):
    print("use model" + model_name)
    model = cPickle.load(open(model_name, 'rb'))

    # X_test = load_all_feature('./RF_feature/test_02.pickle')
    # X_test = pca_transform(X=X_test)
    # Y_test = load_all_labels("./RF_data/label_test_2.txt")
    # print("test ", model.score(X_test, Y_test)*100)

    X_test = load_all_feature(pickle_name)
    Y_test = model.predict(X_test)
    print(Y_test)
    count = 0
    for item in Y_test:
        if int(item) == image_label:
            count = count + 1
    print("predict ", image_label, count)


using_model("./model_saving/model_LinearSVC_16.sav",
            './anh_thuc_te/feature/hoa_bim_bim.pickle', 76)

using_model("./model_saving/model_LinearSVC_16.sav",
            './anh_thuc_te/feature/hoa_cam_chuong.pickle', 31)

using_model("./model_saving/model_LinearSVC_16.sav",
            './anh_thuc_te/feature/hoa_cuc.pickle', 41)

using_model("./model_saving/model_LinearSVC_16.sav",
            './anh_thuc_te/feature/hoa_cuc_van_tho.pickle', 47)

using_model("./model_saving/model_LinearSVC_16.sav",
            './anh_thuc_te/feature/hoa_dai.pickle', 81)

using_model("./model_saving/model_LinearSVC_16.sav",
            './anh_thuc_te/feature/hoa_dam_but.pickle', 83)

using_model("./model_saving/model_LinearSVC_16.sav",
            './anh_thuc_te/feature/hoa_do_quyen.pickle', 72)

using_model("./model_saving/model_LinearSVC_16.sav",
            './anh_thuc_te/feature/hoa_giay.pickle', 95)

using_model("./model_saving/model_LinearSVC_16.sav",
            './anh_thuc_te/feature/hoa_hong.pickle', 74)

using_model("./model_saving/model_LinearSVC_16.sav",
            './anh_thuc_te/feature/hoa_huong_duong.pickle', 54)

using_model("./model_saving/model_LinearSVC_16.sav",
            './anh_thuc_te/feature/hoa_ly.pickle', 6)

using_model("./model_saving/model_LinearSVC_16.sav",
            './anh_thuc_te/feature/hoa_moc_lan.pickle', 87)

using_model("./model_saving/model_LinearSVC_16.sav",
            './anh_thuc_te/feature/hoa_pang_xe.pickle', 52)

using_model("./model_saving/model_LinearSVC_16.sav",
            './anh_thuc_te/feature/hoa_sen.pickle', 78)

using_model("./model_saving/model_LinearSVC_16.sav",
            './anh_thuc_te/feature/hoa_sung.pickle', 73)

using_model("./model_saving/model_LinearSVC_16.sav",
            './anh_thuc_te/feature/hoa_thien_dieu.pickle', 8)

using_model("./model_saving/model_LinearSVC_16.sav",
            './anh_thuc_te/feature/hoa_tra.pickle', 96)

using_model("./model_saving/model_LinearSVC_16.sav",
            './anh_thuc_te/feature/hoa_trang_nguyen.pickle', 44)


# def make_new_model(path):
#     svc_training(path)
#     using_model(path)


# make_new_model(path_to_model_svc)


def RandomForest_training():
    X_train_init = load_all_feature('./RF_feature/train.pickle')
    X_train_mirror = load_all_feature("./RF_feature/train_mirror.pickle")
    X_train = np.concatenate((X_train_init, X_train_mirror), axis=0)

    Y_train = generate_label() + generate_label_mirror()

    clf = RandomForestClassifier(
        n_estimators=600,
        # random_state=0,
        criterion='entropy',
        # min_samples_split=10,
        bootstrap=False,
        # max_leaf_nodes=7
        # class_weight="balanced"
    )
    print("training RF")
    clf.fit(X_train, Y_train)
    print("save model RF")
    filename = './model_saving/model_rf_1.sav'
    cPickle.dump(clf, open(filename, 'wb'))


def using_RF_model(model_name):
    # X_test = load_all_feature('./RF_feature/valid.pickle')
    # Y_test = generate_label()

    X_test = load_all_feature('./RF_feature/test.pickle')
    Y_test = load_all_labels("./RF_data/label_test.txt")

    print("use model RF " + model_name)
    model = cPickle.load(open(model_name, 'rb'))

    result = model.predict(X_test)
    count = 0
    for i in range(len(result)):
        if result[i] == Y_test[i]:
            count = count + 1
    print(float(count)*100 / len(result))


# RandomForest_training()

# using_RF_model('./model_saving/model_rf_1.sav')


def knn():
    # X_train_init = load_all_feature('./RF_feature/train.pickle')
    # X_train_mirror = load_all_feature("./RF_feature/train_mirror.pickle")
    # X_train_val = load_all_feature("./RF_feature/valid.pickle")
    # X_train = np.concatenate(
    #     (X_train_init, X_train_mirror, X_train_val), axis=0)
    # # X_train = load_all_feature('./RF_feature/train.pickle')
    # Y_train = generate_label() + generate_label_mirror() + generate_label()

    X_train = load_all_feature('./RF_feature/test.pickle')
    Y_train = load_all_labels("./RF_data/label_test.txt")

    print("start KNN")
    knn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    knn.fit(X_train, Y_train)

    # validation
    X_val = load_all_feature('./RF_feature/train.pickle')
    Y_val = generate_label()

    # testing data
    # X_val = load_all_feature('./RF_feature/test.pickle')
    # Y_val = load_all_labels("./RF_data/label_test.txt")

    distances, indices = knn.kneighbors(X_val)
    print(indices)

    result = []
    for i in indices:
        result.append(Y_train[i[0]])

    count = 0
    for i in range(len(result)):
        if result[i] == Y_val[i]:
            count = count + 1
    print(float(count)*100 / len(result))


# knn()
