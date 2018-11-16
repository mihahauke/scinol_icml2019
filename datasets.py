#!/usr/bin/env python3

import os
import sys
import pickle
import tarfile
import numpy as np
from mnist import MNIST
from six.moves import urllib
from distributions import normal_scaled, normal_dist_outliers, normal

from sklearn.preprocessing import StandardScaler
# TODO deprecation here
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import sigmoid_kernel

MNIST_DOWNLOAD_DIR = '/tmp/mnist_data/'
MNIST_LECUN_URL = "http://yann.lecun.com/exdb/mnist/"
MNIST_MIRROR_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
MNIST_URL = MNIST_MIRROR_URL

MNIST_TRAIN_IMAGES_FILENAME = 'train-images-idx3-ubyte.gz'
MNIST_TRAIN_LABELS_FILENAME = 'train-labels-idx1-ubyte.gz'
MNIST_TEST_IMAGES_FILENAME = 't10k-images-idx3-ubyte.gz'
MNIST_TEST_LABELS_FILENAME = 't10k-labels-idx1-ubyte.gz'

UCI_DATASETS = "http://archive.ics.uci.edu/ml/machine-learning-databases"
UCI_MADELON = UCI_DATASETS + "/madelon/MADELON"
MADELON_TRAIN = UCI_MADELON + "/madelon_train.data"
MADELON_TRAIN_LABELS = UCI_MADELON + "/madelon_train.labels"
MADELON_TEST = UCI_MADELON + "/madelon_valid.data"
MADELON_TEST_LABELS = UCI_DATASETS + "/madelon/" + "madelon_valid.labels"

UCI_BANK_URL = UCI_DATASETS + "/00222/bank-additional.zip"

UCI_CENSUS_URL = UCI_DATASETS + "/census-income-mld/census.tar.gz"

UCI_COVTYPE_URL = UCI_DATASETS + "/covtype/covtype.data.gz"

CIFAR_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR_DOWNLOAD_DIR = '/tmp/cifar10_data'
CIFAR_EXTRACT_PATH = 'cifar-10-batches-py'
CIFAR_DATA_SHAPE = (32, 32, 3)
CIFAR_CLASSESS_NUM = 10
MNIST_DATA_SHAPE = (28, 28, 1)
MNIST_CLASSES_NUM = 10


def _to_one_hot(int_labels):
    enc = OneHotEncoder(categories='auto')
    return enc.fit_transform(int_labels.reshape([-1, 1])).toarray()


# TODO some regression tasks?
class _Dataset():
    def __init__(self,
                 name,
                 train_data,
                 test_data,
                 input_shape,
                 num_outputs,
                 train_batchsize=None,
                 test_batchsize=None,
                 one_hot=True,
                 seed=None,
                 **kwargs):
        if num_outputs == 2:
            one_hot = False
            num_outputs = 1

        self._name = name
        self._input_shape = list(input_shape)
        self._outputs_num = num_outputs
        self.test = list(test_data)
        self.train = list(train_data)

        self.one_hot = one_hot

        if self.one_hot:
            self.train[1] = _to_one_hot(self.train[1])
            self.test[1] = _to_one_hot(self.test[1])
        self.train_batchsize = train_batchsize
        self.test_batchsize = test_batchsize

        self.seeds = seed

    def get_name(self):
        return self._name

    @property
    def outputs_num(self):
        return self._outputs_num

    @property
    def num_records(self):
        return len(self.train[0]) + len(self.test[0])

    @property
    def input_shape(self):
        return self._input_shape

    def train_batches(self, batchsize=None):
        if batchsize is None:
            batchsize = self.train_batchsize

        x, y = self.train
        num_examples = len(x)

        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        for ai in range(0, num_examples, batchsize):
            bi = min(ai + batchsize, num_examples)
            minibatch = x[perm[ai:bi]], y[perm[ai:bi]]
            yield minibatch

    def test_batches(self, batchsize=None):
        raise NotImplementedError()

    def get_test_data(self):
        return self.test

    def maybe_download(self, url, download_path):
        os.makedirs(download_path, exist_ok=True)

        filename = url.split('/')[-1]
        filepath = os.path.join(download_path, filename)
        if not os.path.exists(filepath):
            # TODO use tqdm
            def _progress(count, block_size, total_size):
                sys.stdout.write('\rDownloading %s %.1f%%' % (filename,
                                                              float(count * block_size) / float(
                                                                  total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
            statinfo = os.stat(filepath)
            print()
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')


class Cifar10(_Dataset):
    def __init__(self,
                 name="cifar10",
                 *args, **kwargs):
        print("Loading cifar10 dataset.")
        self.maybe_download_and_extract()
        train_data, test_data = self.load_dataset()
        super(Cifar10, self).__init__(name,
                                      train_data=train_data,
                                      test_data=test_data,
                                      input_shape=CIFAR_DATA_SHAPE,
                                      num_outputs=CIFAR_CLASSESS_NUM,
                                      *args, **kwargs)

    def maybe_download_and_extract(self):
        """Download and extract the tarball from Alex's website."""
        if not os.path.exists(CIFAR_DOWNLOAD_DIR):
            os.makedirs(CIFAR_DOWNLOAD_DIR)
        filename = CIFAR_URL.split('/')[-1]
        filepath = os.path.join(CIFAR_DOWNLOAD_DIR, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\rDownloading %s %.1f%%' % (filename,
                                                              float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(CIFAR_URL, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        extracted_dir_path = os.path.join(CIFAR_DOWNLOAD_DIR, CIFAR_EXTRACT_PATH)
        if not os.path.exists(extracted_dir_path):
            tarfile.open(filepath, 'r:gz').extractall(CIFAR_DOWNLOAD_DIR)

    def load_dataset(self):
        train_filenames = [os.path.join(CIFAR_DOWNLOAD_DIR,
                                        CIFAR_EXTRACT_PATH,
                                        'data_batch_{}'.format(i))
                           for i in range(1, 6)]
        test_filename = os.path.join(CIFAR_DOWNLOAD_DIR,
                                     CIFAR_EXTRACT_PATH,
                                     'test_batch')
        train_images = []
        train_labels = []

        def process_images(ims):
            ims = ims.reshape([-1, 3, 32, 32]).astype(np.float32) / 255
            return np.transpose(ims, [0, 2, 3, 1])

        for filename in train_filenames:
            with open(filename, 'rb') as file:
                data = pickle.load(file, encoding='latin1')
                images = data['data']
                labels = data['labels']
                train_images.append(images)
                train_labels.append(labels)
        train_images = process_images(np.concatenate(train_images))
        train_labels = np.concatenate(train_labels).astype(np.int64)

        with open(test_filename, 'rb') as file:
            test_data = pickle.load(file, encoding='latin1')
            test_images = test_data['data']
            test_labels = test_data['labels']

        test_images = process_images(test_images)
        test_labels = np.int64(test_labels)

        return (train_images, train_labels), (test_images, test_labels)


class _Penn(_Dataset):
    def __init__(self,
                 name,
                 seed=None,
                 test_ratio=0.33,
                 *args, **kwargs):
        if seed is not None:
            raise NotImplementedError()

        # print("Fetching '{}' dataset. It may take a while.".format(name))
        download_path = "/tmp/penn_{}".format(name)
        os.makedirs(download_path, exist_ok=True)

        x, y = fetch_data(name, return_X_y=True, local_cache_dir=download_path)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, random_state=seed)
        num_outputs = len(np.unique(y))
        super(_Penn, self).__init__("Penn_" + name,
                                    train_data=(x_train, y_train),
                                    test_data=(x_test, y_test),
                                    input_shape=[x.shape[1]],
                                    num_outputs=num_outputs,
                                    seed=seed,
                                    *args, **kwargs)


class UCI_Madelon(_Dataset):
    def __init__(self,
                 name="UCI_Madelon",
                 *args, **kwargs):
        # print("Fetching Madelon dataset. It may take a while.")
        download_path = "/tmp/uci_madelon"

        def download_and_extract(url):
            self.maybe_download(url, download_path)
            file = os.path.join(download_path, url.split("/")[-1])
            return np.genfromtxt(file, delimiter=" ")

        x_train = download_and_extract(MADELON_TRAIN)
        x_test = download_and_extract(MADELON_TEST)
        y_train = (download_and_extract(MADELON_TRAIN_LABELS) + 1) / 2
        y_test = (download_and_extract(MADELON_TEST_LABELS) + 1) / 2

        num_outputs = 2  # len(np.unique(y))
        super(UCI_Madelon, self).__init__(name,
                                          train_data=(x_train, y_train),
                                          test_data=(x_test, y_test),
                                          input_shape=[x_train.shape[1]],
                                          num_outputs=num_outputs,
                                          *args, **kwargs)


class UCI_Bank(_Dataset):
    def __init__(self,
                 name="UCI_Bank",
                 test_ratio=0.33,
                 seed=None,
                 *args, **kwargs):
        # print("Fetching Bank dataset. It may take a while.")
        download_path = "/tmp/uci_bank"

        self.maybe_download(UCI_BANK_URL, download_path)
        file = os.path.join(download_path, UCI_BANK_URL.split("/")[-1])
        import zipfile
        zip_ref = zipfile.ZipFile(file, 'r')
        zip_ref.extractall(download_path)
        zip_ref.close()

        import pandas as pd
        csv_file = os.path.join(download_path, "bank-additional/bank-additional-full.csv")
        dataframe = pd.read_csv(csv_file, delimiter=";")

        y = np.zeros_like(dataframe["y"], dtype=np.int32)
        y[dataframe["y"] == "yes"] = 1
        dataframe.drop("y", axis=1, inplace=True)
        dataframe = pd.get_dummies(dataframe, drop_first=True)
        x = np.float32(dataframe.values)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, random_state=seed)

        num_outputs = 2  # len(np.unique(y))
        super(UCI_Bank, self).__init__(name,
                                       train_data=(x_train, y_train),
                                       test_data=(x_test, y_test),
                                       input_shape=[x_train.shape[1]],
                                       num_outputs=num_outputs,
                                       *args, **kwargs)


class UCI_Census(_Dataset):
    def __init__(self,
                 name="UCI_Census",
                 test_ratio=0.33,
                 seed=None,
                 *args, **kwargs):
        # print("Fetching Census dataset. It may take a while.")
        download_path = "/tmp/uci_census"

        self.maybe_download(UCI_CENSUS_URL, download_path)

        targzfile = os.path.join(download_path, UCI_CENSUS_URL.split("/")[-1])

        tar = tarfile.open(targzfile, mode="r")
        tar.extractall(download_path)
        tar.close()
        os.chmod(download_path + "/census-income.names", 0o770)
        os.chmod(download_path + "/census-income.data", 0o770)
        os.chmod(download_path + "/census-income.test", 0o770)
        import pandas as pd

        train_x = pd.read_csv(download_path + "/census-income.data", header=None, delimiter=",")
        test_x = pd.read_csv(download_path + "/census-income.test", header=None, delimiter=",")

        labels_column = 41
        x = pd.concat([train_x, test_x], axis=0)
        y = np.zeros_like(x[labels_column], dtype=np.int32)
        y[x[labels_column] == " 50000+."] = 1
        x.drop(labels_column, axis=1, inplace=True)

        x_object = x.select_dtypes(include=['object']).copy()
        x_numerical = x.select_dtypes(exclude=['object']).copy()

        x_object = pd.get_dummies(x_object, drop_first=True)

        x = pd.concat([x_numerical, x_object], axis=1)
        x = np.float32(x.values)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, random_state=seed)
        num_outputs = 2  # len(np.unique(y))
        super(UCI_Census, self).__init__(name,
                                         train_data=(x_train, y_train),
                                         test_data=(x_test, y_test),
                                         input_shape=[x_train.shape[1]],
                                         num_outputs=num_outputs,
                                         *args, **kwargs)


class UCI_Covertype(_Dataset):
    def __init__(self,
                 name="UCI_Covertype",
                 test_ratio=0.33,
                 seed=None,
                 *args, **kwargs):
        # print("Fetching Census dataset. It may take a while.")
        download_path = "/tmp/uci_covertype"

        self.maybe_download(UCI_COVTYPE_URL, download_path)

        file = os.path.join(download_path, UCI_COVTYPE_URL.split("/")[-1])

        import gzip
        import pandas as pd
        gzfile = gzip.open(file, 'rb')
        x = pd.read_csv(gzfile, delimiter=",", header=None)

        label_column = 54
        y = np.int32(x[label_column].values)
        x.drop(label_column, axis=1, inplace=True)
        x = np.float32(x.values)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, random_state=seed)

        num_outputs = len(np.unique(y))
        super(UCI_Covertype, self).__init__(name,
                                            train_data=(x_train, y_train),
                                            test_data=(x_test, y_test),
                                            input_shape=[x_train.shape[1]],
                                            num_outputs=num_outputs,
                                            *args, **kwargs)


class Mnist(_Dataset):
    def __init__(self, *args, **kwargs):
        mnist_files = [MNIST_TRAIN_IMAGES_FILENAME,
                       MNIST_TRAIN_LABELS_FILENAME,
                       MNIST_TEST_IMAGES_FILENAME,
                       MNIST_TEST_LABELS_FILENAME]

        for filename in mnist_files:
            self.maybe_download(MNIST_URL + filename, MNIST_DOWNLOAD_DIR)

        # print("Loading mnist data ...")
        mnist_loader = MNIST(MNIST_DOWNLOAD_DIR)
        mnist_loader.gz = True
        train_images, train_labels = mnist_loader.load_training()
        test_images, test_labels = mnist_loader.load_testing()
        process_images = lambda im: (np.array(im).astype(np.float32) / 255.0).reshape((- 1, 28, 28, 1))

        train_images = process_images(train_images)
        test_images = process_images(test_images)
        train_labels = np.int64(train_labels)
        test_labels = np.int64(test_labels)

        super(Mnist, self).__init__(name="mnist",
                                    train_data=(train_images, train_labels),
                                    test_data=(test_images, test_labels),
                                    input_shape=MNIST_DATA_SHAPE,
                                    num_outputs=MNIST_CLASSES_NUM,
                                    *args, **kwargs)


class Synthetic(_Dataset):
    def __init__(self,
                 name="normal",
                 size=100000,
                 num_features=10,
                 test_ratio=0.33,
                 seed=None,
                 distribution=normal_scaled,
                 **kwargs):
        x, labels = distribution(
            size,
            num_features,
            loc=0,
            seed=seed)

        x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=test_ratio, random_state=seed)

        super(Synthetic, self).__init__(
            name=name,
            train_data=(x_train, y_train),
            test_data=(x_test, y_test),
            input_shape=[num_features],
            num_outputs=2,
            **kwargs)


class SyntheticStandardized(Synthetic):
    def __init__(self,
                 name="normal_standardized",
                 **kwargs):
        super(SyntheticStandardized, self).__init__(
            name=name,
            **kwargs)
        scaler = StandardScaler()
        self.train[0] = scaler.fit_transform(self.train[0])
        self.test[0] = scaler.transform(self.test[0])


SynthNormal = lambda **kwargs: Synthetic(
    name="normal",
    size=100000,
    distribution=normal,
    **kwargs)
SynthScaled = lambda **kwargs: Synthetic(
    name="scaled",
    distribution=normal_scaled,
    **kwargs)
SynthOutliers = lambda **kwargs: Synthetic(
    name="outliers",
    distribution=normal_dist_outliers,
    **kwargs)

PennPoker = lambda **kwargs: _Penn("poker", **kwargs)
PennFars = lambda **kwargs: _Penn("fars", **kwargs)
PennKddcup = lambda **kwargs: _Penn("kddcup", **kwargs)
PennConnect4 = lambda **kwargs: _Penn("connect-4", **kwargs)
PennShuttle = lambda **kwargs: _Penn("shuttle", **kwargs)
PennSleep = lambda **kwargs: _Penn("sleep", **kwargs)
