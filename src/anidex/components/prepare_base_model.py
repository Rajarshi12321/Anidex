import warnings

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import itertools
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.metrics import classification_report, f1_score, confusion_matrix


# Tensorflow Libraries
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models, Model

# from keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.layers.experimental import preprocessing
# from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

from pathlib import Path

import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf

from anidex.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.applications.EfficientNetB3(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top,
            pooling=self.config.params_pooling,
        )

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        # Data Augmentation Step
        augment = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.15),
                tf.keras.layers.RandomZoom(0.15),
                tf.keras.layers.RandomContrast(0.15),
            ],
            name="AugmentationLayer",
        )

        inputs = layers.Input(shape=(224, 224, 3), name="inputLayer")
        x = augment(inputs)

        pretrain_out = model(x, training=False)
        x = layers.Dense(256)(pretrain_out)
        x = layers.Activation(activation="relu")(x)
        x = BatchNormalization()(x)
        x = layers.Dropout(0.45)(x)
        x = layers.Dense(classes)(x)

        outputs = layers.Activation(
            activation="softmax", dtype=tf.float32, name="activationLayer"
        )(
            x
        )  # mixed_precision need separated Dense and Activation layers
        full_model = Model(inputs=inputs, outputs=outputs)

        # flatten_in = tf.keras.layers.Flatten()(model.output)
        # prediction = tf.keras.layers.Dense(
        #     units=classes,
        #     activation="softmax"
        # )(flatten_in)

        # full_model = tf.keras.models.Model(
        #     inputs=model.input,
        #     outputs=prediction
        # )

        # full_model.compile(
        #     optimizer=Adam(0.0005),
        #     loss='categorical_crossentropy',
        #     metrics=['accuracy']
        # )

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        print(full_model.summary())

        return full_model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate,
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
