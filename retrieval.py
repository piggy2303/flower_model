from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import numpy as np
import cPickle
from sklearn import metrics
from sklearn.model_selection import train_test_split

print("Load feature")
all_feature = open("./allFeature.pickle", "rb")
all_feature_data = cPickle.load(all_feature)
print("loading label")
all_label = []
with open("labels.txt", "r") as all_Label_file:
    for i in all_Label_file:
        all_label.append(i.rstrip())
# print all_label

print("start KNN")          

test = cPickle.load(open("./test23.pickle", "rb"))
knn = NearestNeighbors(n_neighbors=10, algorithm='ball_tree')
knn.fit(all_feature_data, all_label)
distances, indices = knn.kneighbors(test)
print indices
for i in indices[0]:
    print all_label[i]