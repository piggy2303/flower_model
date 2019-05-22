from sklearn.model_selection import cross_val_score
import numpy as np
import cPickle
from sklearn.preprocessing import normalize

# chia bo du lieu flower detect theo 

def load_all_feature(path_to_feature):
    print("Load feature", path_to_feature)
    all_feature = cPickle.load(open(path_to_feature, "rb"))
    all_feature_norm = normalize(all_feature, norm='l2')
    return all_feature_norm

def load_label():
    label_flower = np.array([1]*5132)
    label_non_flower = np.array([0]*5158)
    label = np.concatenate((label_flower, label_non_flower), axis=None)
    return label


flower_train = load_all_feature("./feature/flower_train.pickle")
flower_val = load_all_feature("./feature/flower_validation.pickle")
flower_test = load_all_feature("./feature/flower_test.pickle")

non_flower_train = load_all_feature("./feature/nonflower_train.pickle")
non_flower_val = load_all_feature("./feature/nonflower_validation.pickle")
non_flower_test = load_all_feature("./feature/nonflower_test.pickle")


X = np.concatenate((flower_train,flower_test,flower_val, non_flower_train,non_flower_test,non_flower_val), axis=0)
Y = load_label()

model = cPickle.load(open("./model_detect/model_1.sav", 'rb'))

scores = cross_val_score(model, X,Y,cv=5)

print (scores)                                       
