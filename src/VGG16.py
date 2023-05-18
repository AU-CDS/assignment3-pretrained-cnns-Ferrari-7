'''
Note to Ross: 
I have unfortunately not finished testing the code yet, but I'm working on it.

'''

# Importing packages
import os
import pandas as pd
# tf tools
import tensorflow as tf
# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
# cifar10 data - 32x32
from tensorflow.keras.datasets import cifar10
# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model
# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD
#scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
# for plotting
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    # converting the metadata into Pandas objects
    test_metadata = pd.read_json(os.path.join("..", "images", "metadata", "test_data.json"), lines = True)
    train_metadata = pd.read_json(os.path.join("..", "images", "metadata", "train_data.json"), lines = True)
    val_metadata = pd.read_json(os.path.join("..", "images", "metadata", "val_data.json"), lines = True)

    # !!! TAKING A SMALL FRACTION FOR TESTING PURPOSES
    train_metadata = train_metadata.sample(frac=0.01)
    test_metadata = test_metadata.sample(frac=0.01)
    val_metadata = val_metadata.sample(frac=0.01)

    # changing the column with the image path from a relative path to an absolute path
    test_metadata["image_path"] = "/work/" + test_metadata["image_path"]
    train_metadata["image_path"] = "/work/" + train_metadata["image_path"]
    val_metadata["image_path"] = "/work/" + val_metadata["image_path"]

    # Defining data generater
    # flip along x axis (mirror image)
    datagen = ImageDataGenerator(horizontal_flip=True, 
                                rotation_range=20,
                                validation_split=0.1)

    # splitting the data into train, test and validation by using "flow_from_dataframe"
    # source: code adapted from Kaggle-user vencerlanz09 
    # (link to source: https://www.kaggle.com/code/vencerlanz09/indo-fashion-classification-using-efficientnetb0)
    BATCH_SIZE = 32
    TARGET_SIZE = (224, 224)
    train_images = datagen.flow_from_dataframe(
        dataframe=train_metadata,
        #directory = os.path.join("..", "images", "train"),
        x_col='image_path',
        y_col='class_label',
        target_size=TARGET_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
        subset='training')

    val_images = datagen.flow_from_dataframe(
        dataframe=val_metadata,
        #directory = os.path.join("..", "images", "val"),
        x_col='image_path',
        y_col='class_label',
        target_size=TARGET_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42)

    test_images = datagen.flow_from_dataframe(
        dataframe=test_metadata,
        #directory = os.path.join("..", "images", "test"),
        x_col='image_path',
        y_col='class_label',
        target_size=TARGET_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=False)

    # assigning labels by getting unique labels from the class column
    label_names = test_metadata['class_label'].unique()
    
    return label_names, train_images, val_images, test_images, test_metadata

def load_model():
    # load model without classifier layers
    model = VGG16(include_top=False, 
                pooling='avg',
                input_shape=(32, 32, 3))

    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
        
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    bn = BatchNormalization()(flat1) # NTS : removed one layer
    class1 = Dense(128, 
                activation='relu')(bn)
    output = Dense(15, 
                activation='softmax')(class1) # NTS: changed from 10 to 15 (number of classes)

    # define new model
    model = Model(inputs=model.inputs, 
                outputs=output)

    # compile
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9)
    sgd = SGD(learning_rate=lr_schedule)

    model.compile(optimizer=sgd,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    return model


def train_clf(model, train_images, val_images):
    H = model.fit(train_images, 
            validation_data = val_images,
            batch_size=128,
            epochs=2, # CHANGED FROM 10 FOR TESTING
            verbose=1)

    return H, model

# Using plot_history() function to see how model performs during training.
def plot_history(H, epochs):
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    # save the plot
    plt.savefig(os.path.join("out", "history_plt.png"))


def clf_report(model, test_images, test_metadata, label_names):
    predictions = model.predict(test_images, batch_size=128)
    predictions = np.argmax(predictions, axis=1)
    # generating predictions
    predictions = [label_names[k] for k in predictions] # NTS: NEW
    y_test = list(test_metadata.class_label) # (new)
    clf_report = classification_report(y_test, # removed: .argmax(axis=1)
                            predictions, # removed: .argmax(axis=1)
                            target_names=label_names)
    # clf_report = print(clf_report)
    # save the classification report
    txtfile_path = os.path.join("out", "clf_report.txt")
    txtfile = open(txtfile_path, "w")
    txtfile.write(str(clf_report))
    txtfile.close


def main():
    label_names, train_images, val_images, test_images, test_metadata = load_data()
    model = load_model()
    H, model = train_clf(model, train_images, val_images)
    plot_history(H, 2) # CHANGED FROM 10 FOR TESTING
    clf_report(model, test_images, test_metadata, label_names)

if __name__=="__main__":
    main()

