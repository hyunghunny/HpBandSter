"""
Worker for Example 5 - Keras
============================

In this example implements a small CNN in Keras to train it on MNIST.
The configuration space shows the most common types of hyperparameters and
even contains conditional dependencies.

We'll optimise the following hyperparameters:

+-------------------------+----------------+-----------------+------------------------+
| Parameter Name          | Parameter type |  Range/Choices  | Comment                |
+=========================+================+=================+========================+
| Learning rate           |  float         | [1e-6, 1e-2]    | varied logarithmically |
+-------------------------+----------------+-----------------+------------------------+
| Optimizer               | categorical    | {Adam, SGD }    | discrete choice        |
+-------------------------+----------------+-----------------+------------------------+
| SGD momentum            |  float         | [0, 0.99]       | only active if         |
|                         |                |                 | optimizer == SGD       |
+-------------------------+----------------+-----------------+------------------------+
| Number of conv layers   | integer        | [1,3]           | can only take integer  |
|                         |                |                 | values 1, 2, or 3      |
+-------------------------+----------------+-----------------+------------------------+
| Number of filters in    | integer        | [4, 64]         | logarithmically varied |
| the first conf layer    |                |                 | integer values         |
+-------------------------+----------------+-----------------+------------------------+
| Number of filters in    | integer        | [4, 64]         | only active if number  |
| the second conf layer   |                |                 | of layers >= 2         |
+-------------------------+----------------+-----------------+------------------------+
| Number of filters in    | integer        | [4, 64]         | only active if number  |
| the third conf layer    |                |                 | of layers == 3         |
+-------------------------+----------------+-----------------+------------------------+
| Dropout rate            |  float         | [0, 0.9]        | standard continuous    |
|                         |                |                 | parameter              |
+-------------------------+----------------+-----------------+------------------------+
| Number of hidden units  | integer        | [8,256]         | logarithmically varied |
| in fully connected layer|                |                 | integer values         |
+-------------------------+----------------+-----------------+------------------------+

Please refer to the compute method below to see how those are defined using the
ConfigSpace package.

The network does not achieve stellar performance when a random configuration is samples,
but a few iterations should yield an accuracy of >90%. To speed up training, only
8192 images are used for training, 1024 for validation.
The purpose is not to achieve state of the art on MNIST, but to show how to use
Keras inside HpBandSter, and to demonstrate a more complicated search space.
"""
import os

try:
    import keras
    from keras import regularizers
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    from keras import backend as K
except:
    raise ImportError("For this example you need to install keras.")

try:
    import torchvision
    import torchvision.transforms as transforms
except:
    raise ImportError("For this example you need to install pytorch-vision.")



import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

import logging
logging.basicConfig(level=logging.DEBUG)
from lookup import *
import time



class KerasWorker(Worker):
    def __init__(self, N_train=55000, N_valid=5000, gpu_id=0, **kwargs):
        super().__init__(**kwargs)

        # TODO:read surrogates from lookup directory
        self.lookup = load('data2', data_folder='./lookup/')

        # Set using single GPU only
        os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        self.batch_size = 64

        img_rows = 28
        img_cols = 28
        self.num_classes = 10

        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            self.input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            self.input_shape = (img_rows, img_cols, 1)


        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        # zero-one normalization
        x_train /= 255
        x_test /= 255


        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)


        self.x_train, self.y_train = x_train[:N_train], y_train[:N_train]
        self.x_validation, self.y_validation = x_train[-N_valid:], y_train[-N_valid:]
        self.x_test, self.y_test   = x_test, y_test

        self.input_shape = (img_rows, img_cols, 1)




    def compute(self, config, budget, working_directory, *args, **kwargs):
        """
        Simple example for a compute function using a feed forward network.
        It is trained on the MNIST dataset.
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """
        try:
            num_epochs = budget
            target_lookup_index = kwargs["lookup_index"]["lookup_index"]
            hpv = self.lookup.get_hyperparam_vectors()[target_lookup_index]
            test_err = self.lookup.get_test_errors(num_epochs)[target_lookup_index]
            elapsed_time = self.lookup.get_elapsed_times(num_epochs)[target_lookup_index]

            return ({
                'loss': test_err, # remember: HpBandSter always minimizes!
                'info': {
                             'test accuracy': 1.0 - test_err,
                             'elapsed time': elapsed_time,
                             'lookup index': int(target_lookup_index)
                        }
            })

        except:

            start_time = time.time()
            model = Sequential()

            model.add(Conv2D(int(config['c1_depth']), kernel_size=(int(config['window_size']),int(config['window_size'])),
                             activation='relu',
                             input_shape=self.input_shape))
            model.add(MaxPooling2D(pool_size=(int(config['p1_size']), int(config['p1_size']))))


            model.add(Conv2D(config['c2_depth'], kernel_size=(int(config['window_size']),int(config['window_size'])),
                             activation='relu',
                             input_shape=self.input_shape))
            model.add(MaxPooling2D(pool_size=(int(config['p2_size']), int(config['p2_size']))))



            model.add(Dropout(float(1.0 - config['keep_prop_rate'])))
            model.add(Flatten())

            model.add(Dense(config['f1_width'], activation='relu',
                            kernel_regularizer=keras.regularizers.l2(config['reg_param'])))
            model.add(Dropout(float(1.0 - config['keep_prop_rate'])))
            model.add(Dense(self.num_classes, activation='softmax'))


            optimizer = keras.optimizers.Adam(lr=config['learning_rate'])

            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=optimizer,
                          metrics=['accuracy'])

            model.fit(self.x_train, self.y_train,
                      batch_size=self.batch_size,
                      epochs=int(budget),
                      verbose=0,
                      validation_data=(self.x_test, self.y_test))

            train_score = model.evaluate(self.x_train, self.y_train, verbose=0)
            val_score = model.evaluate(self.x_validation, self.y_validation, verbose=0)
            test_score = model.evaluate(self.x_test, self.y_test, verbose=0)

            #import IPython; IPython.embed()
            return ({
                'loss': 1-val_score[1], # remember: HpBandSter always minimizes!
                'info': {	'test accuracy': test_score[1],
                            'elapsed time': time.time() - start_time,
                          n  'train accuracy': train_score[1],
                            'validation accuracy': val_score[1],
                            'number of parameters': model.count_params(),
                        }

            })


    @staticmethod
    def get_configspace():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """

        cs = CS.ConfigurationSpace()

        learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=1e-4, upper=10**-0.5, default_value=1e-2,
                                                       log=True)
        reg_param = CSH.UniformFloatHyperparameter('reg_param', lower=0.0, upper=1.0, default_value=0.5, log=False)
        keep_prop_rate = CSH.UniformFloatHyperparameter('keep_prop_rate', lower=0.1, upper=1.0, default_value=0.5,
                                                        log=False)
        cs.add_hyperparameters([learning_rate, reg_param, keep_prop_rate])

        c1_depth = CSH.UniformIntegerHyperparameter('c1_depth', lower=1, upper=350, default_value=32, log=False)
        p1_size = CSH.UniformIntegerHyperparameter('p1_size', lower=2, upper=3, default_value=2, log=False)
        c2_depth = CSH.UniformIntegerHyperparameter('c2_depth', lower=1, upper=350, default_value=64, log=False)
        p2_size = CSH.UniformIntegerHyperparameter('p2_size', lower=2, upper=3, default_value=2, log=False)
        window_size = CSH.UniformIntegerHyperparameter('window_size', lower=2, upper=10, default_value=2, log=False)
        f1_width = CSH.UniformIntegerHyperparameter('f1_width', lower=1, upper=1024, default_value=512, log=False)

        cs.add_hyperparameters([c1_depth, p1_size, c2_depth, p2_size, window_size, f1_width])
        return cs


if __name__ == "__main__":
    worker = KerasWorker(run_id='0')
    cs = worker.get_configspace()

    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(config=config, budget=1, working_directory='.')
    print(res)
