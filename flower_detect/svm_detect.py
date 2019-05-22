from sklearn.svm import SVC
import numpy as np
import cPickle
from sklearn.preprocessing import normalize


def load_all_feature(path_to_feature):
    print("Load feature", path_to_feature)
    all_feature = cPickle.load(open(path_to_feature, "rb"))
    all_feature_norm = normalize(all_feature, norm='l2')
    return all_feature_norm


def load_label(arr_flower, arr_non_flower):
    label_flower = np.array([1]*len(arr_flower))
    label_non_flower = np.array([0]*len(arr_non_flower))
    label = np.concatenate((label_flower, label_non_flower), axis=None)
    return label


def svc_training():
    flower_train = load_all_feature("./feature/flower_train.pickle")
    non_flower_train = load_all_feature("./feature/nonflower_train.pickle")

    X_train = np.concatenate((flower_train, non_flower_train), axis=0)
    Y_train = load_label(flower_train, non_flower_train)

    clf = SVC(
        degree=2,
        gamma='auto',
        kernel='linear',
    )

    print("training model")
    clf.fit(X_train, Y_train)
    print("save model")
    filename = './model_detect/model_2.sav'
    cPickle.dump(clf, open(filename, 'wb'))


def using_model(model_name):
    # flower_validation = load_all_feature("../RF_feature/test.pickle")
    flower_validation = load_all_feature("./feature/flower_test.pickle")
    non_flower_validation = load_all_feature("./feature/nonflower_test.pickle")
    print("use model")
    model = cPickle.load(open(model_name, 'rb'))

    actual_1 = model.predict(flower_validation)
    # actual_0 = model.predict(non_flower_validation)

    # print(actual_1)
    # count = 0
    # for item in actual_1:
    #     item = int(item)
    #     if item == 1:
    #         count = count + 1

    # print(float(count)/len(actual_1))
    print(calcu_f1_score(actual_1, actual_0))

# using_model('./model_detect/model_1.sav')


def calcu_f1_score(actual_1, actual_0):
    TP = np.count_nonzero(actual_1)
    FP = np.count_nonzero(actual_0)
    FN = len(actual_1) - TP
    TN = len(actual_0) - FP
    Precision = float(TP) / (TP+FP)
    Recall = float(TP) / (TP+FN)
    F1_Score = 2*(Recall * Precision) / (Recall + Precision)

    return F1_Score


def testing_model(model_name):
    flower_test = load_all_feature("./feature/flower_test.pickle")
    non_flower_test = load_all_feature(
        "./feature/nonflower_test.pickle")
    print("use model")
    model = cPickle.load(open(model_name, 'rb'))

    actual_1 = model.predict(flower_test)
    actual_0 = model.predict(non_flower_test)

    print(calcu_f1_score(actual_1, actual_0))


# svc_training()
# using_model('./model_detect/model_1.sav')
# testing_model('./model_detect/model_2.sav')

testing_model('./model_detect/model_1.sav')


def test():
    arr = np.random.rand(8000, 4000)
    label1 = np.array([[1]*4000 + [0]*4000])

    clf = SVC(gamma='auto')
    print("training model")
    clf.fit(arr, label1[0])

    arr_val = np.random.rand(1, 4000)
    print(clf.predict(arr_val))


# test()
