import importlib
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.resnet50 import ResNet50
from facematch.age_prediction.utils.metrics import earth_movers_distance, age_mae

CLASSES_NUMBER = 100
MOBILENET_MODEL_NAME = "MobileNetV2"
RESNET_MODEL_NAME = "ResNet50"


class AgeClassificationNet:
    def __init__(self, base_model, img_shape):
        self.base_model = base_model
        self.img_shape = img_shape
        self._get_base_module()

    def build(self):
        if self.base_model == MOBILENET_MODEL_NAME:
            base_model = MobileNetV2(input_shape=self.img_shape, include_top=False, weights="imagenet")
        elif self.base_model == RESNET_MODEL_NAME:
            base_model = ResNet50(input_shape=self.img_shape, include_top=False, weights="imagenet")

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(units=CLASSES_NUMBER, activation="softmax")(x)
        self.model = Model(inputs=base_model.input, outputs=x)

    def _get_base_module(self):
        # import Keras base model module
        if self.base_model == MOBILENET_MODEL_NAME:
            self.base_module = importlib.import_module("keras.applications.mobilenet_v2")
        elif self.base_model == RESNET_MODEL_NAME:
            self.base_module = importlib.import_module("keras.applications.resnet50")

    def compile(self):
        learning_rate = 0.001
        optimizer = Adam(lr=learning_rate)

        loss = earth_movers_distance
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[age_mae, "accuracy"])

    def preprocessing_function(self):
        return self.base_module.preprocess_input