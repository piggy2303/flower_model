import scipy.io as sio
import os
from sklearn.preprocessing import normalize
import cPickle

# os.mkdir('images/01')
# os.rename("./text.txt", "./images/01/text.txt")

# for i in range(102):
#     os.mkdir('images/'+str(i+1))

# labels_arr = []
# with open("labels.txt", "r") as labels:
#     for i in labels:
#         i = i.rstrip()
#         labels_arr.append(i)

# # print labels_arr

# with open("list_images.txt", "r") as list_images:
#     count = 0
#     for image_name in list_images:
#         image_name = image_name.rstrip()
#         os.rename("./images/jpg/" + image_name,
#                   "./images/" + labels_arr[count] + "/" + image_name)


#         print("./images/jpg/" + image_name,
#               "./images/" + labels_arr[count] + "/" + image_name)
#         count = count + 1
def slipt_data():
    mat = sio.loadmat('setid.mat')
    print mat['tstid']
    print len(mat['tstid'][0])

    with open("data_test.txt", "wb") as data_test:
        for target_list in mat['tstid'][0]:
            data_test.write(str(target_list) + "\n")

    with open("data_valid.txt", "wb") as data_valid:
        for target_list in mat['valid'][0]:
            data_valid.write(str(target_list) + "\n")

    with open("data_train.txt", "wb") as data_train:
        for target_list in mat['trnid'][0]:
            data_train.write(str(target_list) + "\n")


def check_normalize():
    arr = cPickle.load(open("./feature/1.pickle", "rb"))
    print arr
    arr_normalize = normalize(arr, norm='l2', axis=1,
                              copy=True, return_norm=False)
    print arr_normalize


check_normalize()
