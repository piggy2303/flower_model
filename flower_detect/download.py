import requests
import urllib
import os


def is_downloadable(url):
    try:
        h = requests.get(url, allow_redirects=False, timeout=5)
        status_code = h.status_code
        print status_code
        if status_code == 200:
            if 'text' in h.headers.get('content-type').lower():
                return False
            if 'html' in h.headers.get('content-type').lower():
                return False
            return True
    except:
        return False


file_name = './list_image_room.txt'
folder_name = 'flower'
# for fileTextName in os.listdir(path):
#     print("open file " + fileTextName)

count = 1145

with open(file_name, "r") as text_file:
    for urlRaw in text_file:
        url = urlRaw.rstrip()
        print("visit image " + str(count))
        ImageFilename = "image_" + str(count) + ".jpg"

        if is_downloadable(url):
            r = requests.get(url, allow_redirects=False)
            open("./images/" + folder_name + "/" + ImageFilename,
                 'wb').write(r.content)

            print("done " + str(count))

        else:
            print("error " + str(count))

        count = count + 1
