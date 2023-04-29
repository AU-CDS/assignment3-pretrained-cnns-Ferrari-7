
# Importing packages
import os
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

    # assigning labels
    label_names = ["saree", "women kurta",
                   "leggins and salwar",
                   "palazzo", "lehenga",
                   "dupatta", "blouse",
                   "gown", "dhoti pants",
                   "petticoats", "women mojari",
                   "men kurta", "nehru jacket",
                   "sherwani", "men mojari"]
    
    return label_names, X_train, y_train, X_test, y_test

def load_model():
    # load model without classifier layers
    model = VGG16(include_top=False, 
              pooling='avg',
              input_shape=(32, 32, 3))
    
    # disable training of conv layers
    for layer in model.layers:
    layer.trainable = False


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

