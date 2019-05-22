from sklearn.model_selection import KFold
import numpy as np
import cPickle
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def load_all_feature(select_num, all_feature):
    print("Load feature")
    # all_feature = cPickle.load(open("./allFeature.pickle", "rb"))
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


def cross_val(X, y):
    kf = KFold(n_splits=5, shuffle=True)
    kf.get_n_splits(X)

    arr_accu = []
    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        accu = knn(X_train, X_test, y_train, y_test)
        arr_accu.append(accu)
    sum(arr_accu) / float(len(arr_accu))


def knn(X_train, X_test, y_train, y_test):
    print("start KNN")
    knn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    knn.fit(X_train, y_train)
    count = 0
    accu = 0
    for test_case in X_test:
        distances, indices = knn.kneighbors([test_case])
        if (str(y_train[indices[0]]).split("'")[1] == y_test[count]):
            accu = accu + 1
        count = count + 1

    print float(accu) / count
    return float(accu) / count


all_feature = cPickle.load(open("./allFeature.pickle", "rb"))

X = load_all_feature(2500, all_feature)
y = np.array(load_all_labels())

cross_val(X, y)
