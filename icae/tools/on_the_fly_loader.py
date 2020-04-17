raise DeprecationWarning()
import dask.dataframe as dd
import pandas as pd
import numpy as np
import keras.utils
from icae.tools.config_loader import config
from sklearn.model_selection import train_test_split
import pandas as pd
from joblib import delayed, Parallel


class DataGeneratorAE(keras.utils.Sequence):
    def __init__(self, df_path, indices, batch_size, shuffle=True):
        self.path = df_path
        self.batch_size = batch_size
        self.indices = indices
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.IDs) / self.batch_size))

    # def __getitem__(self, index):
    #    """Generate one batch of data"""
    # Generate indexes of the batch
    #   indexes = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

    # Generate data
    #   X = self.__data_generation(indexes)

    #  return X, X

    def on_epoch_end(self):
        """Shuffels indices after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, indexes):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        tasks = [
            delayed(pd.read_hdf)(self.path, mode="r", start=i, stop=i + 1)
            for i in indexes
        ]
        X = pd.concat(Parallel(n_jobs=12)(tasks))

        return X, X

    def generate_data_chunk(self, indexes):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)

        X = pd.read_hdf(
            self.path, mode="r", start=indexes[0], stop=indexes[0] + len(indexes)
        )

        return X

    def generate_data(self, indexes):
        return self.__data_generation(indexes)[0]


class DataRepresentation:
    def __init__(self, filepath, validatation_split=0.1, batch_size=1000):
        self.batch_size = batch_size
        self.validation_split = validatation_split

        self.table = pd.HDFStore(filepath, mode="r").get_storer(config.data.hdf_key)
        indices = np.arange(self.table.nrows)  # this can be quite big (~300 MB)
        self.train_indices, self.val_indices = train_test_split(
            indices, test_size=validatation_split
        )
        del indices

        self.training_generator = DataGeneratorAE(
            filepath, self.train_indices, batch_size, True
        )
        self.validation_generator = DataGeneratorAE(
            filepath, self.val_indices, batch_size, True
        )

        print("Loaded file with %e rows" % self.table.nrows)

    def get_config(self):
        """a dict to use with `model.fit_generator(**returnValue)`"""
        returnValue = {
            "generator": self.training_generator,
            "validation_data": self.validation_generator,
            #'use_multiprocessing': True,
            #'workers'            : 12
        }
        return returnValue

    def validation_sample(self, size=1000):
        validation = np.random.choice(self.val_indices, size, replace=False)
        return self.validation_generator.generate_data(validation)

    def validation_sample_chunk(self, size=1000):
        validation = np.random.choice(self.val_indices, size, replace=False)
        return self.validation_generator.generate_data_chunk(validation)
