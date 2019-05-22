import pymongo
from bson.json_util import dumps

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["flower"]
mycol = mydb["collection_flower_detail"]


label = [1, 2, 3]
arr_flower = []

for item in label:
    mongo_item = mycol.find_one({'index': item})
    dumps(mongo_item)
    arr_flower.append(dumps(mongo_item))

print(arr_flower)

# with open('./detail.txt', "r") as text_file:
#     arr = []
#     for line in text_file:
#         line = str(line)
#         arr.append(line)
#         if len(line) == 1:
#             myquery = {"index": int(arr[0])}
#             newvalues = {"$set": {"detail": arr}}

#             print myquery
#             print newvalues
#             mycol.update_one(myquery, newvalues)

#             arr = []
# print(len(line))
