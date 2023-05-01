'''
Note to Ross: 
I have unfortunately not finished the code since I'm struggling a bit with figuring out how to load the data.

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
    ((X_train, y_train), (X_test, y_test)) = #load data here...
    

    X_train = X_train.astype("float") / 255.
    X_test = X_test.astype("float") / 255.

    # integers to one-hot vectors
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)

    # deading the metadata into Pandas objects
    test_metadata = pd.read_json(os.path.join("data", "images", "metadata", "test_data.json"))
    train_metadata = pd.read_json(os.path.join("data", "images", "metadata", "train_data.json"))
    val_metadata = pd.read_json(os.path.join("data", "images", "metadata", "val_data.json"))

    # Defining data generater
    # flip along x axis (mirror image)
    datagen = ImageDataGenerator(horizontal_flip=True, 
                                rotation_range=20,
                                validation_split=0.1)

    # splitting the data into train, test and validation 
    # source: code adapted from Kaggle-user vencerlanz09 
    # (link to source: https://www.kaggle.com/code/vencerlanz09/indo-fashion-classification-using-efficientnetb0)
    train_images = datagen.flow_from_dataframe(
        dataframe=train_metadata,
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
        x_col='image_path',
        y_col='class_label',
        target_size=TARGET_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=False)

    # assigning labels by getting unique labels from the class column
    label_names = test_metadata['class_label'].unique()
    
    return label_names, X_train, y_train, X_test, y_test

def load_model():
    # load model without classifier layers
    model = VGG16(include_top=False, 
              pooling='avg',
              input_shape=(32, 32, 3))
    
    # disable training of conv layers
    for layer in model.layers:
    layer.trainable = False

    # Add new classification layers
    # tf.keras.backend.clear_session()
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu')(flat1)
    output = Dense(10, activation='softmax')(class1)

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



def train_clf(model):
    H = model.fit(X_train, y_train, 
            validation_split=0.1,
            batch_size=128,
            epochs=10,
            verbose=1)

    return H

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
    plt.save(os.path.join("out", "history_plt.png"))

def clf_report(model, X_test, y_test, label_names):
    predictions = model.predict(X_test, batch_size=128)
    clf_report = print(classification_report(y_test.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=label_names))
    
    # save the classification report
    txtfile_path = os.path.join("out", "clf_report.txt")
    txtfile = open(txtfile_path, "w")
    txtfile.write(clf_report)
    txtfile.close


def main():
    load_data
    train_clf
    plot_history
    clf_report

if __name__=="__main__":
    main()

