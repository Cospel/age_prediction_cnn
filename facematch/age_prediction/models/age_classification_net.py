import importlib
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.mobilenet_v2 import MobileNetV2
from keras_applications.resnet_v2 import ResNet50V2
from keras.layers.core import Dropout

from facematch.age_prediction.utils.utils import age_ranges_number
from facematch.age_prediction.utils.metrics import earth_movers_distance, age_mae

AGES_NUMBER = 100
AGE_RANGES_NUMBER = 10
RANGE_LENGTH = 5
AGE_RANGES_UPPER_THRESH = 80
GENDERS_NUMBER = 2
MOBILENET_MODEL_NAME = "MobileNetV2"
RESNET_MODEL_NAME = "ResNet50"


class AgeClassificationNet:
    def __init__(self, base_model_name, img_shape, learning_rate, range_mode=False, predict_gender=False):
        self.base_model_name = base_model_name
        self.img_shape = img_shape
        self.range_mode = range_mode
        self.predict_gender = predict_gender
        self.learning_rate = learning_rate
        self._get_base_module()

    def build(self):
        if self.base_model_name == MOBILENET_MODEL_NAME:
            print("Mobilenetv2 build")
            self.base_model = MobileNetV2(input_shape=self.img_shape, include_top=False, weights="imagenet")
        elif self.base_model_name == RESNET_MODEL_NAME:
            self.base_model = ResNet50(input_shape=self.img_shape, include_top=False, weights="imagenet")

        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)

        if self.range_mode:
            age_classes_number = age_ranges_number()
            age_output = Dense(units=age_classes_number, activation="softmax", name="age_output")(x)
        else:
            age_output = Dense(units=AGES_NUMBER, activation="softmax", name="age_output")(x)

        if not self.predict_gender:
            self.model = Model(inputs=self.base_model.input, outputs=age_output)
        else:
            gender_output = Dense(units=GENDERS_NUMBER, activation="softmax", name="gender_output")(x)
            self.model = Model(inputs=self.base_model.input, outputs=[age_output, gender_output])

        # list indices of base model layers
        # for i, layer in enumerate(self.base_model.layers):
        #     print("{} {}".format(i, layer.__class__.__name__))

    def _get_base_module(self):
        # import Keras base model module
        if self.base_model_name == MOBILENET_MODEL_NAME:
            print("Mobilenetv2 base")
            self.base_module = importlib.import_module("keras.applications.mobilenet_v2")
        elif self.base_model_name == RESNET_MODEL_NAME:
            # resnet_v2 has same preprocessing as mobilenetv2
            self.base_module = importlib.import_module("keras.applications.mobilenet_v2")

    def compile(self):
        optimizer = Adam(lr=self.learning_rate)

        if self.range_mode:
            age_loss = "categorical_crossentropy"
            age_metrics = ["accuracy"]
        else:
            age_loss = earth_movers_distance
            age_metrics = [age_mae, "accuracy"]

        if not self.predict_gender:
            self.model.compile(optimizer=optimizer, loss=age_loss, metrics=age_metrics)
        else:
            self.model.compile(
                optimizer=optimizer,
                loss={"age_output": age_loss, "gender_output": "categorical_crossentropy"},
                loss_weights={"age_output": 1.0, "gender_output": 1.0},
                metrics={"age_output": age_metrics, "gender_output": "accuracy"},
            )

    def preprocessing_function(self):
        return self.base_module.preprocess_input
