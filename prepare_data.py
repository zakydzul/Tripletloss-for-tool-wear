from pandas import read_csv
import numpy as np
import os
from scipy import interpolate
from sklearn.preprocessing import minmax_scale
from sklearn.utils import shuffle
import tensorflow as tf

class preprocessing():
    def __init__(self, pattern, mother_path=None):
        self.mother_path = mother_path
        self.file_pattern = pattern
        self.data =[]
        self.label =[]
        self.l=0
        for x in self.file_pattern:
            temp = read_csv(os.path.join(self.mother_path, x), header=None)
            self.data.append(np.array(temp, dtype=np.float32))
            self.label.append(np.full(len(temp), self.l))
            self.l += 1

    def concate(self):
        self.data = np.concatenate((self.data), axis=0)
        self.label = np.concatenate((self.label), axis=0)
        return self.data, self.label

    def fft_preprocessing(self, ar):
        L = ar.shape[1];
        Fs = 25600;
        t = np.linspace(0.0, L, L) / Fs
        f = np.linspace(0.0, L // 2, L // 2) * Fs / L
        l = []
        for i in range(ar.shape[0]):
            Y1 = np.fft.fftn(ar[i])
            P1 = 2 * np.abs(Y1 / L)
            x_fft = np.square(P1[0:L // 2])
            p = interpolate.interp1d(f, x_fft.flatten())
            x_new = np.arange(0, f[-1], 2)
            y_new = p(x_new)
            l.append(y_new)
        l = np.array(l)
        l = minmax_scale(l, axis=1)
        return np.asarray(l)

    def tensor_conv(self, data, label,samp_num = 0.7, BATCH_SIZE = 64,SHUFFLE_BUFFER_SIZE = 1000):
        data, label = shuffle(data, label)

        samp = round(samp_num * data.shape[0])
        train_examples = tf.convert_to_tensor(data[:samp], dtype=tf.float16)
        train_labels = tf.convert_to_tensor(label[:samp], dtype=tf.int8)
        test_examples = tf.convert_to_tensor(data[samp:], dtype=tf.float16)
        test_labels = tf.convert_to_tensor(label[samp:], dtype=tf.int8)

        train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

        BATCH_SIZE = BATCH_SIZE
        SHUFFLE_BUFFER_SIZE = SHUFFLE_BUFFER_SIZE

        train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        test_dataset = test_dataset.batch(BATCH_SIZE)

        return train_dataset, test_dataset
