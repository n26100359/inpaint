import os
import cv2
import numpy as np
import csv
# this function is for read image,the input is directory name
mini = 999
def read_directory(directory_name):
    array_of_img = [] # this if for store all of the image data
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(directory_name):
        #print(filename) #just for test
        #img is used to store the image data
        img = cv2.imread(directory_name + "/" + filename)
        array_of_img.append(img)
        #print(img)
    return array_of_img

def cutimage(setname, minimum, dataset):
    global mini
    croppeds = []

    if "train" == setname:
        mini = min([min(k.shape[:2]) for k in dataset])

    for i in range(len(dataset)):
        img = dataset[i]
        width, height = dataset[i].shape[:2]
        if width > height:
            x1 = (width - height)//2
            x2 = (width + height)//2
            cropped = img[x1:x2]
#            print(cropped.shape)
        else:
            y1 = (height - width)//2
            y2 = (height + width)//2
            cropped = img[:, y1:y2]
#           print(cropped.shape)
        res = cv2.resize(cropped, dsize=(96, 96), interpolation=cv2.INTER_CUBIC)
#       print(res.shape)
        croppeds.append(res)
    print(len(croppeds))
    x = []
    y = []
    outer=[]
    for i in range(len(dataset)):
        middle = croppeds[i][int(96*0.25):int(96*0.75), int(96*0.25):int(96*0.75)]
        y.append(middle)
    np.save(f"y{setname}96_mid.npy", y)
    for i in range(len(dataset)):
        outer.append(croppeds[i])
        outer[i][int(96*0.25):int(96*0.75), int(96*0.25):int(96*0.75)] = 0
        x.append(outer[i])

    np.save(f"x{setname}96_mid.npy", x)

if "__main__" == __name__:
#    read_directory("dataset_updated/training_set/drawings")
    cutimage("train", mini, read_directory("dataset_updated/training_set/drawings"))
#    read_directory("dataset_updated/validation_set/drawings")
    cutimage("test", mini, read_directory("dataset_updated/validation_set/drawings"))
