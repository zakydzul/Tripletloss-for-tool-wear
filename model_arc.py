import tensorflow as tf
import tensorflow_addons as tfa


class model():
    def __init__(self, input_size=0, num_units=None, DO_val=None, metric=None):
        self.input = input_size
        self.HP_NUM_UNITS = num_units
        self.HP_DROPOUT = DO_val
        self.METRIC_ACCURACY = metric

    def model_res_net(self,hparams):
        input1 = tf.keras.layers.Input(shape=self.input.shape)
        reshape = tf.keras.layers.Reshape((self.input.shape[0], 1))(input1)
        b1_cnv1d_1 = tf.keras.layers.Conv1D(filters=32, kernel_size=20,padding='SAME', activation='relu')(reshape)
        b1_bn_1 = tf.keras.layers.BatchNormalization()(b1_cnv1d_1)
        b1_LRL_1 = tf.keras.layers.LeakyReLU(alpha=0.3)(b1_bn_1)
        b1_out = tf.keras.layers.MaxPooling1D(pool_size=2)(b1_LRL_1)
        #--------#
        b2_cnv1d_1=tf.keras.layers.Conv1D(filters=32, kernel_size=20, padding='SAME', activation='relu')(b1_out)
        b2_bn_1=tf.keras.layers.BatchNormalization()(b2_cnv1d_1)
        b2_LRL_1=tf.keras.layers.LeakyReLU(alpha=0.3)(b2_bn_1)
        b2_out=tf.keras.layers.MaxPooling1D(pool_size=1)(b2_LRL_1)
        #--------#
        b2_add = tf.keras.layers.Add()([b1_out,b2_out])
        #--------#
        b3_cnv1d_1 = tf.keras.layers.Conv1D(filters=64, kernel_size=10,padding='SAME', activation='relu')(b2_add)
        b3_bn_1 = tf.keras.layers.BatchNormalization()(b3_cnv1d_1)
        b3_LRL_1 = tf.keras.layers.LeakyReLU(alpha=0.2)(b3_bn_1)
        b3_out = tf.keras.layers.MaxPooling1D(pool_size=2)(b3_LRL_1)
        #-------#
        b4_cnv1d_1 = tf.keras.layers.Conv1D(filters=64, kernel_size=10, padding='SAME', activation='relu')(b3_out)
        b4_bn_1 = tf.keras.layers.BatchNormalization()(b4_cnv1d_1)
        b4_LRL_1 = tf.keras.layers.LeakyReLU(alpha=0.2)(b4_bn_1)
        b4_out = tf.keras.layers.MaxPooling1D(pool_size=1)(b4_LRL_1)
        #-------#
        b4_add = tf.keras.layers.Add()([b3_out,b4_out])
        do_FCN_1=tf.keras.layers.Dropout(0.3)(b4_add)
        fltn_FCN=tf.keras.layers.Flatten()(do_FCN_1)
        dense_FCN_1=tf.keras.layers.Dense(hparams[self.HP_NUM_UNITS], activation='relu')(fltn_FCN)
        do_FCN_2=tf.keras.layers.Dropout(hparams[self.HP_DROPOUT])(dense_FCN_1)
        dense_FCN_2=tf.keras.layers.Dense(256, activation=None,
                              kernel_regularizer=tf.keras.regularizers.L2(1e-3),
                              kernel_initializer=tf.keras.initializers.HeUniform())(do_FCN_2)
        # No activation on final dense layer
        lambda_out=tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(dense_FCN_2)

        model = tf.keras.models.Model(inputs=input1, outputs=lambda_out)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0001),
            loss=tfa.losses.TripletSemiHardLoss())
        return model

    def model_seq(self,hparams):
        model = tf.keras.Sequential([
            tf.keras.layers.Reshape((self.input.shape[0], 1), input_shape=self.input.shape),
            tf.keras.layers.Conv1D(filters=32, kernel_size=20, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.3),
            tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
            # -----------
            tf.keras.layers.Conv1D(filters=32, kernel_size=20, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.3),
            tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
            # -----------
            tf.keras.layers.Conv1D(filters=64, kernel_size=10, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
            # -----------
            tf.keras.layers.Conv1D(filters=64, kernel_size=10, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
            # -----------
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(hparams[self.HP_NUM_UNITS], activation='relu'),
            tf.keras.layers.Dropout(hparams[self.HP_DROPOUT]),
            tf.keras.layers.Dense(256, activation=None,
                                  kernel_regularizer=tf.keras.regularizers.L2(1e-3),
                                  kernel_initializer=tf.keras.initializers.HeUniform()),
            # No activation on final dense layer
            tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))  # L2 normalize embeddings
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0001),
            loss=tfa.losses.TripletSemiHardLoss())
        return model
