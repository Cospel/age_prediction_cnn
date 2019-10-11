import os
import cv2
import numpy as np
import keras
from keras.utils import to_categorical
from imgaug import augmenters as iaa
from facematch.age_prediction.utils.utils import build_age_vector, age_ranges_number, get_age_range_index

AGES_NUMBER = 100
GENDERS_NUMBER = 3
MAX_AGE = 100


class DataGenerator(keras.utils.Sequence):
    """inherits from Keras Sequence base object"""

    def __init__(self, args, samples_directory, basemodel_preprocess, generator_type, shuffle):
        self.samples_directory = samples_directory
        self.model_type = args["type"]
        self.base_model = args["base_model"]
        self.basemodel_preprocess = basemodel_preprocess
        self.batch_size = args["batch_size"]
        self.sample_files = []
        self.img_dims = (args["img_dim"], args["img_dim"])  # dimensions that images get resized into when loaded
        self.age_deviation = args["age_deviation"]
        self.predict_gender = args["predict_gender"] if "predict_gender" in args else False
        self.range_mode = args["range_mode"] if "range_mode" in args else False
        self.age_classes_number = age_ranges_number() if self.range_mode else AGES_NUMBER
        self.dataset_size = None
        self.generator_type = generator_type
        self.shuffle = shuffle

        self.load_sample_files()
        self.indexes = np.arange(self.dataset_size)

        self.on_epoch_end()  # for training data: call ensures that samples are shuffled in first epoch if shuffle is set to True

    def __len__(self):
        return int(np.ceil(self.dataset_size / self.batch_size))  #  number of batches per epoch

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]  # get batch indexes
        list_ids = [i for i in batch_indexes]

        X, y_age, y_gender = self.__data_generator(list_ids)

        X = self.augmentor(X)
        if not self.predict_gender:
            return X, y_age
        else:
            return X, [y_age, y_gender]

    def on_epoch_end(self):
        self.indexes = np.arange(self.dataset_size)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generator(self, list_ids):
        # initialize images
        X, Y_AGE, Y_GENDER = [], [], []

        for i in list_ids:
            x, y_age, y_gender = self.process_file(self.sample_files[i])

            X.append(x)
            Y_AGE.append(y_age)
            Y_GENDER.append(y_gender)

        return np.asarray(X), np.asarray(Y_AGE), np.asarray(Y_GENDER)

    def augmentor(self, images):
        "Apply data augmentation"
        seq = iaa.Sequential(
            [iaa.Fliplr(0.5), iaa.GaussianBlur((0, 0.5))], random_order=True  # horizontally flip 50% of all images
        )
        return seq.augment_images(images)

    def process_file(self, file_name):
        image, y_age, y_gender = None, None, None

        # Load image
        file_path = os.path.join(self.samples_directory, file_name)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, self.img_dims)

        # apply basenet specific preprocessing
        image = self.basemodel_preprocess(image)

        # Obtain age
        age = int(file_name.split("_")[0])
        age = min(age, MAX_AGE)

        # Save AGE label and image to training dataset
        if self.model_type == "classification":
            if self.range_mode:
                range_index = get_age_range_index(age)
                # transform label to categorical vector
                y_age = to_categorical(range_index, self.age_classes_number)
            else:
                # Build AGE vector
                age_vector = build_age_vector(age, self.age_deviation)
                y_age = age_vector
        else:
            age = float(age / MAX_AGE)
            y_age = age

        if self.predict_gender:
            gender = int(file_name.split("_")[1])
            # transform label to categorical vector
            y_gender = to_categorical(gender, 2)

        return image, y_age, y_gender

    def load_sample_files(self):
        """
        Loads file names of training samples
        :return:
        """
        self.sample_files = [f for f in os.listdir(self.samples_directory) if (f.endswith("JPG") or f.endswith("jpg"))]

        self.dataset_size = len(self.sample_files)
