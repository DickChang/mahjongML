import os, os.path
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import math
import numpy as np
from tensorflow import keras
import tensorflow as tf

test_percent = 0.1 # Use this percent of our sample data for testing
class_names = [] # this gets populated in load_data()

# all images will be 3-channel, float colors in the range [0,1]
def load_data():
    num_files_loaded = 0
    all_filenames = []
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    data_dir = './images/mahjong/data_format/'
    all_filenames = os.listdir(data_dir)

    # partition and count all the input files by type
    tile_accumulator = {}
    for file in all_filenames:
        file_split = file.split('_')
        tile_type = file_split[0] + "_" + file_split[1]
        if tile_type not in tile_accumulator:
            tile_accumulator[tile_type] = 1
        else:
            tile_accumulator[tile_type] += 1

    # create the class names
    tile_accumulator = OrderedDict(sorted(tile_accumulator.items()))
    class_names.extend(list(tile_accumulator.keys()))

    # Load every file and divide them between the training set and the test set
    # scale jpg images to the [0,1] range
    for tile_type in class_names:
        tile_type_imgs = []
        tile_type_filenames = [x for x in all_filenames if (tile_type+"_") in x]
        for filename in tile_type_filenames:
            num_files_loaded += 1
            if num_files_loaded % 100 == 0:
                print("Images loaded... " + str(num_files_loaded))
            loaded_image = mpimg.imread( data_dir + filename ) # numpy.array
            if '.png' in filename:
                #loaded_image = loaded_image.tolist()
                loaded_image = loaded_image
            elif '.jpg' in filename:
                #loaded_image = (loaded_image/255).tolist() # scale jpg images
                loaded_image = loaded_image/255 # scale jpg images
            #tile_type_imgs.append(loaded_image.tolist())
            tile_type_imgs.append(loaded_image)
            #print(filename + ": " + str(np.array(tile_type_imgs[-1]).shape))

        #print(tile_type_filenames)
        #print(tile_type + ":" + str(len(tile_type_imgs)))

        num_test_images = math.ceil(tile_accumulator[tile_type] * test_percent)
        for i in range(num_test_images):
            test_images.append(tile_type_imgs.pop( random.randint(0, len(tile_type_imgs)-1) ))
            test_labels.append(class_names.index(tile_type))

        train_images.extend(tile_type_imgs)
        for i in range(len(tile_type_imgs)):
            train_labels.append(class_names.index(tile_type))

    print("Loaded " + str(num_files_loaded) + " images.")
    #print("Training images shape: " + str(np.array(train_images).shape))
    #print("Training Labels shape: " + str(np.array(train_labels).shape))
    #print("Test images shape: " + str(np.array(test_images).shape))
    #print("Test Labels shape: " + str(np.array(test_labels).shape))
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    print("Training images shape: " + str(train_images.shape))
    print("Training labels shape: " + str(train_labels.shape))
    print("Test images shape: " + str(test_images.shape))
    print("Test labels shape: " + str(test_labels.shape))
    return (train_images, train_labels), (test_images, test_labels)

def plot_single(image, filename):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)
    plt.savefig(filename)
    plt.show()

if __name__ == "__main__":       
    # import data set and define string labels
    (train_images, train_labels), (test_images, test_labels) = load_data()
    # print(class_names)

    plot_single( train_images[20], "train_images_20.png" )

    # plot the first 25 training images
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.savefig('train_images_first25.png')
    plt.show()

    if keras.backend.image_data_format() == 'channels_first':
        #print("channels first")
        my_shape = (3, None, None)
    elif keras.backend.image_data_format() == 'channels_last':
        #print("channels last")
        my_shape = (None, None, 3)
    else:
        print("Unknown keras.backend.image_data_format(): " + keras.backend.image_data_format())
        exit(0)

    # setup model layers
    model = keras.Sequential([
        #keras.layers.Input(shape=my_shape),
        keras.layers.Conv2D(16, (4,4), activation='elu', input_shape=my_shape),
        keras.layers.Conv2D(32, (4,4), activation='elu'),
        keras.layers.Dropout(0.5),
        keras.layers.Conv2D(64, (4,4), activation='elu'),
        keras.layers.Dropout(0.5),
        keras.layers.Conv2D(128, (1,1)),
        keras.layers.GlobalMaxPooling2D(),
        #keras.layers.Flatten(),
        keras.layers.Dense(len(class_names), activation='softmax')
    ])

    # compile the model with loss function, optimizer, and metrics
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Feed the training data to the model - in this example, the train_images and train_labels arrays.
    # The model learns to associate images and labels.
    # We ask the model to make predictions about a test set - in this example, the test_images array
    #   We verify that the predictions match the labels from the test_labels array.
    model.fit(train_images, train_labels, epochs=5, verbose=1)

    # Evaluate accuracy
    # compare how the model performs on the test dataset
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)

    # make predictions
    predictions = model.predict(test_images)

    print("Exiting.")
    exit(0)

    # look at 0th image and the prediction array
    plot_image_and_values(0, predictions, test_labels, test_images)

    # Plot the first X test images, their predicted label, and the true label
    # Color correct predictions in blue, incorrect predictions in red
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions, test_labels)
    plt.savefig('predictions_first15.png')
    plt.show
