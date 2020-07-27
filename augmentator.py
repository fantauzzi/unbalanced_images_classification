import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import time
import PIL.Image as Image
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, \
    plot_precision_recall_curve, f1_score, confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.initializers import constant
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pathlib import Path
from tensorflow.keras.applications.densenet import DenseNet121

IMAGE_SIZE = (224, 224)
IMAGE_SHAPE = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
EPOCHS = 40
BATCH_SIZE = 64
VALIDATION_BATCH_SIZE = 64
THETA = .5
DATASET_ROOT = '/home/fanta/.keras/datasets/102flowers/jpg'
CHECKPOINTS_DIR = 'checkpoints'
CHECKPOINTS_PATH = CHECKPOINTS_DIR + '/weights.{epoch:05d}.hdf5'
count_to_be_dropped = 0
use_extended_dataset = False
NP_SEED = 31
TF_SEED = 32
AUGMENTATION = 2
TARGET_SUBDIR = 'augmented'
TARGET_DIR = DATASET_ROOT + '/' + TARGET_SUBDIR
metadata_file_name = 'augmented.csv'


def main():
    np.random.seed(NP_SEED)
    tf.random.set_seed(TF_SEED)

    def load_dataset_2(file_name):
        file_names_df = pd.read_csv(file_name)

        ''' Map the class labels to strings '1' and '0' because scikit-learn train_test_split() requires label classes
         to be strings, in order to split the dataset with stratification (stratification throws an exception if class 
         labels are numbers). '''

        file_names_df['class'] = file_names_df['multi_class'].map(lambda x: 1 if x == 51 else 0)

        return file_names_df

    file_names_df = load_dataset_2('flower_classes.csv')

    file_names_df = file_names_df[file_names_df['class'] == 1]
    print('Loaded information for', len(file_names_df), 'files with positive samples.')
    print('Making', AUGMENTATION, 'new samples for every positive sample.')

    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=90,
                                                                      shear_range=.2,
                                                                      zoom_range=(.75, 1.20),
                                                                      brightness_range=(.7, 1.2))

    # TODO try other interpolations, e.g. bicubic
    generated_data = image_generator.flow_from_dataframe(dataframe=file_names_df,
                                                         directory=DATASET_ROOT,
                                                         save_to_dir=TARGET_DIR,
                                                         target_size=IMAGE_SIZE,
                                                         x_col='file_name',
                                                         y_col='class',
                                                         class_mode='raw',
                                                         batch_size=BATCH_SIZE,
                                                         shuffle=False)
    wanted_batches = int(AUGMENTATION * np.ceil(generated_data.samples / generated_data.batch_size))
    generated_images_count = 0
    for image_batch, label_batch in generated_data:
        assert np.sum(label_batch) == len(image_batch)
        generated_images_count += len(image_batch)
        print('.', end='', flush=True)
        if generated_data.total_batches_seen == wanted_batches:
            break
    print('\nGenerated', generated_images_count, 'images in', TARGET_DIR)
    with open(metadata_file_name, 'w') as target_file:
        target_file.write('file_name,label,class\n')
        for file_name in Path(TARGET_DIR).glob('*.png'):
            target_file.write(TARGET_SUBDIR + '/' + str(file_name.name) + ',51\n')
    print('Written metadata file', metadata_file_name)

if __name__ == '__main__':
    main()
