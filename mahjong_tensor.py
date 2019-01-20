import os, os.path
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import math
import numpy as np

test_percent = 0.1 # Use this percent of our sample data for testing
class_names = [] # this gets populated in load_data()

def load_data():
    num_files_loaded = 0
    all_filenames = []
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    data_dir = './images/mahjong/data_format/'
    all_filenames = os.listdir(data_dir)

    tile_accumulator = {}
    for file in all_filenames:
        file_split = file.split('_')
        tile_type = file_split[0] + "_" + file_split[1]
        if tile_type not in tile_accumulator:
            tile_accumulator[tile_type] = 1
        else:
            tile_accumulator[tile_type] += 1
    tile_accumulator = OrderedDict(sorted(tile_accumulator.items()))
    class_names.extend(list(tile_accumulator.keys()))

    for tile_type in class_names:
        tile_type_imgs = []
        tile_type_filenames = [x for x in all_filenames if (tile_type+"_") in x]
        for filename in tile_type_filenames:
            num_files_loaded += 1
            tile_type_imgs.append(mpimg.imread( data_dir + filename ))
            #print(filename + ": " + str(np.array(tile_type_imgs[-1]).shape))

        #print(tile_type_filenames)
        #print(tile_type + ":" + str(len(tile_type_imgs)))

        num_test_images = math.ceil(tile_accumulator[tile_type] * test_percent)
        for i in range(num_test_images):
            test_data.append(tile_type_imgs.pop( random.randint(0, len(tile_type_imgs)-1) ))
            test_labels.append(class_names.index(tile_type))

        train_data.extend(tile_type_imgs)
        for i in range(len(tile_type_imgs)):
            train_labels.append(class_names.index(tile_type))

    print("Loaded " + str(num_files_loaded) + " files.")
    print("Training data shape: " + str(np.array(train_data).shape))
    print("Training Labels shape: " + str(np.array(train_labels).shape))
    print("Test Data shape: " + str(np.array(test_data).shape))
    print("Test Labels shape: " + str(np.array(test_labels).shape))
    return (train_data, train_labels), (test_data, test_labels)

if __name__ == "__main__":       
    load_data()
    print(class_names)

#img = mpimg.imread('./images/mahjong/data_format/bamboo_1_0000019.jpg')

#print(img)
#print(img.shape)

# plot the first training image and save to png
#plt.figure()
#plt.imshow(img)
#plt.colorbar()
#plt.grid(False)
#plt.savefig('plot.png')
#plt.show()

