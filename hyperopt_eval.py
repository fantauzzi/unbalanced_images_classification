import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from matplotlib import use, is_interactive

use('TkAgg')
# plt.ion()
print('Using', plt.get_backend(), 'as graphics backend.')
print('Is interactive:', is_interactive())
import random
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, f1_score, confusion_matrix, \
    precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.initializers import constant
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pathlib import Path
from tensorflow.keras.applications.densenet import DenseNet121
from hyperopt import hp, STATUS_OK
from hyperopt.pyll.stochastic import sample
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin


def run_experiment(params):
    n_epochs = params['n_epochs']
    """batch_size = params['batch_size']
    val_batch_size = params['val_batch_size']
    test_set_fraction = params['test_set_fraction']
    augm_batch_size = params['augm_batch_size']
    augm_factor = params['augm_factor']
    theta = params['theta']
    image_size = params['image_size']
    image_shape = params['image_shape']
    dataset_root = params['dataset_root']
    checkpoints_dir = params['checkpoints_dir']
    checkpoints_path = params['checkpoints_path']
    augm_target_subdir = params['augm_target_subdir']
    augm_target_dir = params['augm_target_dir']
    aug_metadata_file_name = params['aug_metadata_file_name']
    py_seed = params['py_seed']
    np_seed = params['np_seed']
    tf_seed = params['tf_seed']
    """

    # np.random.seed(np_seed)
    # tf.random.set_seed(tf_seed)
    # random.seed(py_seed)

    test_set_fraction = params['test_set_fraction']
    batch_size = params['batch_size']
    augm_factor = params['augm_factor']


    loss = 100 * np.abs(.25 - test_set_fraction) + (3 - augm_factor) ** 2 + 10/576*(batch_size-24)**2 + random.uniform(-.0015, .0015)
    return {'loss': loss, 'params': params, 'status': STATUS_OK}


if __name__ == '__main__':

    params = {'n_epochs': 2,  # TODO use a named tuple instead?
              'batch_size': 24,
              'val_batch_size': 64,
              'test_set_fraction': .2,
              'augm_batch_size': 64,
              'augm_factor': 3,
              'theta': .5,
              'image_shape': (224, 224, 3),
              'dataset_root': '/home/fanta/.keras/datasets/102flowers/jpg',
              'checkpoints_dir': 'checkpoints',
              'augm_target_subdir': 'augmented',
              'aug_metadata_file_name': 'augmented.csv',
              'py_seed': 44,
              'np_seed': 43,
              'tf_seed': 42}

    params['checkpoints_path'] = params['checkpoints_dir'] + '/weights.{epoch:05d}.hdf5'
    params['augm_target_dir'] = params['dataset_root'] + '/' + params['augm_target_subdir']
    params['image_size'] = params['image_shape'][:2]
    hyper_space = {'batch_size': hp.quniform('batch_size', 16, 32, 1),
                   'test_set_fraction': hp.uniform('test_set_fraction', .15, .35),
                   'augm_factor': hp.choice('augm_factor', (0, 1, 2, 3, 4)),
                   'n_epochs': 2}

    tpe_algorithm = tpe.suggest
    bayes_trials = Trials()
    best = fmin(fn=run_experiment, space=hyper_space, algo=tpe.suggest, max_evals=100, trials=bayes_trials)
    print(bayes_trials.best_trial)

    ...
    # input('Press [Enter] to end.')
