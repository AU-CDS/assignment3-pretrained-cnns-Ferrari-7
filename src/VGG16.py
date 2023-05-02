'''
Note to Ross: 
I have unfortunately not finished the code since I'm struggling a bit with fitting the model to the generated data.
(it gives me the following error on line 140: "ValueError: Asked to retrieve element 0, but the Sequence has length 0")

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
    # !!! NTS remember to change ".." to "data" in final version
    test_metadata = pd.read_json(os.path.join("..", "images", "metadata", "test_data.json"))
    train_metadata = pd.read_json(os.path.join("..", "images", "metadata", "train_data.json"))
    val_metadata = pd.read_json(os.path.join("..", "images", "metadata", "val_data.json"))

    # !!!
    train_metadata.sample(frac=0.05)

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
    
    return label_names, train_images, val_images, test_images

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
    bn = BatchNormalization()(flat1)
    class1 = Dense(256, 
                activation='relu')(bn)
    class2 = Dense(128, 
                activation='relu')(class1)
    output = Dense(10, 
                activation='softmax')(class2)

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
            epochs=10,
            verbose=1)

    return H, epochs

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

def clf_report(model, test_images, label_names):
    predictions = model.predict(test_images, batch_size=128)
    clf_report = print(classification_report(test_images.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=label_names))
    
    # save the classification report
    txtfile_path = os.path.join("out", "clf_report.txt")
    txtfile = open(txtfile_path, "w")
    txtfile.write(clf_report)
    txtfile.close


def main():
    label_names, train_images, val_images, test_images = load_data()
    model = load_model()
    H, epochs = train_clf(model, train_images, val_images)
    plot_history(H, epochs)
    clf_report(model, label_names, test_images)

if __name__=="__main__":
    main()

