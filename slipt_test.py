import cPickle
import numpy as np


all_feature = cPickle.load(open("./RF_feature/test.pickle", "rb"))
print(all_feature)

print(all_feature[1])

test_02 = [all_feature[1]]

for i in range(3, 6149, 1):
    if i % 2 == 1:
        test_02 = np.concatenate((test_02, [all_feature[i]]), axis=0)

print(test_02)
cPickle.dump(test_02, open("./RF_feature/test_02.pickle", "wb"))

print(len(test_02))

print(all_feature[1])
print(all_feature[3])
print(all_feature[5])
print(test_02[0])
print(test_02[1])
print(test_02[2])
