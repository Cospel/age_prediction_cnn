import importlib
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.resnet50 import ResNet50
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout

CLASSES_NUMBER = 100
GENDERS_NUMBER = 2
MOBILENET_MODEL_NAME = "MobileNetV2"
RESNET_MODEL_NAME = "ResNet50"


class AgeRegressionNet:
    def __init__(self, base_model, img_shape, learning_rate, predict_gender=False):
        self.base_model_name = base_model
        self.img_shape = img_shape
        self.predict_gender = predict_gender
        self.learning_rate = learning_rate
        self._get_base_module()

    def build(self):
        if self.base_model_name == MOBILENET_MODEL_NAME:
            self.base_model = MobileNetV2(input_shape=self.img_shape, include_top=False, weights="imagenet")
        elif self.base_model_name == RESNET_MODEL_NAME:
            self.base_model = ResNet50(input_shape=self.img_shape, include_top=False, weights="imagenet")

        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        if not self.predict_gender:
            x = Dense(units=1, activation="sigmoid")(x)
            self.model = Model(inputs=self.base_model.input, outputs=x)
        else:
            age_output = Dense(units=1, activation="sigmoid", name="age_output")(x)
            gender_output = Dense(units=GENDERS_NUMBER, activation="softmax", name="gender_output")(x)
            self.model = Model(inputs=self.base_model.input, outputs=[age_output, gender_output])

    def _get_base_module(self):
        # import Keras base model module
        if self.base_model_name == MOBILENET_MODEL_NAME:
            self.base_module = importlib.import_module("keras.applications.mobilenet_v2")
        elif self.base_model_name == RESNET_MODEL_NAME:
            self.base_module = importlib.import_module("keras.applications.resnet50")

    def compile(self):
        optimizer = Adam(lr=self.learning_rate)

        age_loss = "mean_squared_error"
        if not self.predict_gender:
            self.model.compile(optimizer=optimizer, loss=age_loss)
        else:
            self.model.compile(
                optimizer=optimizer,
                loss={"age_output": age_loss, "gender_output": "categorical_crossentropy"},
                loss_weights={"age_output": 1.0, "gender_output": 1.0},
                metrics={"gender_output": "accuracy"},
            )

    def preprocessing_function(self):
        return self.base_module.preprocess_input
