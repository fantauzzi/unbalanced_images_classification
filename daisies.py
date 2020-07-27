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
BATCH_SIZE = 24
VALIDATION_BATCH_SIZE = 64
THETA = .5
DATASET_ROOT = '/home/fanta/.keras/datasets/102flowers/jpg'
CHECKPOINTS_DIR = 'checkpoints'
CHECKPOINTS_PATH = CHECKPOINTS_DIR + '/weights.{epoch:05d}.hdf5'
count_to_be_dropped = 0
use_extended_dataset = False
NP_SEED = 31
TF_SEED = 32


def plot_metrics(history):
    epsilon = 1e-7
    epochs = [i + 1 for i in history.epoch]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    metrics = ['loss', 'auc', 'precision', 'recall']
    fig, axs = plt.subplots(3, 2)
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plot_row = n // 2
        plot_col = n - plot_row * 2
        axs[plot_row, plot_col].grid(True, axis='x' if metric == 'loss' else 'both')
        lns1 = axs[plot_row, plot_col].plot(epochs, history.history[metric], color=colors[0], linestyle="--",
                                            label='Train')
        if metric != 'loss':
            axs[plot_row, plot_col].plot(epochs, history.history['val_' + metric], color=colors[0], label='Val')
        axs[plot_row, plot_col].set_xlabel('Epoch')
        axs[plot_row, plot_col].set_ylabel(name)
        if metric == 'loss':
            loss_min, loss_max = min(history.history['loss']), max(history.history['loss'])
            loss_range = loss_max - loss_min
            val_loss_min, val_loss_max = min(history.history['val_loss']), max(history.history['val_loss'])
            val_loss_range = val_loss_max - val_loss_min
            gap = .05
            axs[plot_row, plot_col].set_ylim([min(history.history['loss']) - loss_range * gap,
                                              max(history.history['loss']) + loss_range * gap])
            ax2 = axs[plot_row, plot_col].twinx()
            ax2.set_ylim([min(history.history['val_loss']) - val_loss_range * gap,
                          max(history.history['val_loss']) + val_loss_range * gap])
            ax2.set_ylabel('Validation Loss')
            lns2 = ax2.plot(epochs, history.history['val_' + metric], color=colors[0], label='Val')
            lns = lns1 + lns2
            labs = [l.get_label() for l in lns]
            ax2.legend(lns, labs, loc=0)
        elif metric == 'auc':
            axs[plot_row, plot_col].set_ylim([0.8, 1])
        else:
            axs[plot_row, plot_col].set_ylim([0, 1])
        axs[plot_row, plot_col].set_xticks(epochs)
        if metric != 'loss':
            axs[plot_row, plot_col].legend()
    plot_row, plot_col = 2, 0
    precision = np.array(history.history['precision'])
    recall = np.array(history.history['recall'])
    precision_val = np.array(history.history['val_precision'])
    recall_val = np.array(history.history['val_recall'])
    train_F1 = 2. * (precision * recall) / (precision + recall + epsilon)
    val_F1 = 2. * (precision_val * recall_val) / (precision_val + recall_val + epsilon)
    axs[plot_row, plot_col].grid(True)
    axs[plot_row, plot_col].plot(epochs, train_F1, color=colors[0], linestyle="--", label='Train')
    axs[plot_row, plot_col].plot(epochs, val_F1, color=colors[0], label='Val')
    axs[plot_row, plot_col].set_xlabel('Epoch')
    axs[plot_row, plot_col].set_ylabel('F1')
    axs[plot_row, plot_col].set_ylim([0, 1])
    axs[plot_row, plot_col].set_xticks(epochs)
    axs[plot_row, plot_col].legend()
    fig.subplots_adjust(wspace=.3, hspace=.3)


def plot_classified_samples(validation_samples, model=None, theta=THETA):
    for image_batch, label_batch in validation_samples:
        print("Image batch shape: ", image_batch.shape)
        print("Label batch shape: ", label_batch.shape)
        break

    if model is not None:
        predicted_batch = model.predict(image_batch)
        predicted_id = (np.squeeze(predicted_batch) >= theta).astype(int)
        predicted_label_batch = np.array(['daisy' if item == 1 else 'not daisy' for item in predicted_id])
        label_id = label_batch.astype(int)

    plt.figure(figsize=(10, 9))
    plt.subplots_adjust(hspace=0.5)
    for n in range(30):
        plt.subplot(6, 5, n + 1)
        plt.imshow(image_batch[n])
        if model is not None:
            color = "green" if predicted_id[n] == label_id[n] else "red"
            plt.title(predicted_label_batch[n], color=color)
        plt.axis('off')
    _ = plt.suptitle("Model predictions (green: correct, red: incorrect)")


def plot_misclassified_samples(validation_samples, model, theta=THETA):
    max_to_be_plotted = 30
    to_be_plotted = []
    to_be_plotted_label = []
    # Find the first max_to_be_plotted images that are misclassified by the model
    steps = np.ceil(validation_samples.samples / validation_samples.batch_size)
    for image_batch, label_batch in validation_samples:
        if validation_samples.total_batches_seen > steps:
            break
        predicted_batch = model.predict(image_batch)
        predicted_batch_id = (np.squeeze(predicted_batch) >= theta).astype(int)
        label_batch_id = label_batch.astype(int)
        correct_label_batch = np.array(['daisy' if item == 1 else 'not daisy' for item in label_batch_id])
        assert len(image_batch) == len(predicted_batch_id)
        assert len(correct_label_batch) == len(image_batch)
        assert len(predicted_batch_id) == len(label_batch_id)
        for i in range(len(image_batch)):
            if predicted_batch_id[i] != label_batch_id[i]:
                to_be_plotted.append(image_batch[i])
                to_be_plotted_label.append(correct_label_batch[i])
                if len(to_be_plotted) == max_to_be_plotted:
                    break
        if len(to_be_plotted) == max_to_be_plotted:
            break

    plt.figure(figsize=(10, 9))
    plt.subplots_adjust(hspace=0.5)
    for n in range(len(to_be_plotted)):
        plt.subplot(6, 5, n + 1)
        plt.imshow(to_be_plotted[n])
        plt.title(to_be_plotted_label[n])
        plt.axis('off')
    _ = plt.suptitle("Sample of misclassified images with their correct classification")


def plot_auc_and_pr(t_y, p_y):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

    fpr, tpr, thresholds = roc_curve(t_y, p_y, pos_label=1)
    ax = axs[0, 0]
    ax.grid()
    ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % ('Daisies', auc(fpr, tpr)))
    ax.legend()
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    precision, recall, thresholds = precision_recall_curve(t_y, p_y)
    ax = axs[1, 0]
    ax.set_xlim([-.05, 1.05])
    ax.set_ylim([-.05, 1.05])
    ax.grid()
    ax.plot(precision, recall, label='%s (AP Score:%0.2f)' % ('Daisies', average_precision_score(t_y, p_y)))
    ax.legend()
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')

    thresholds = np.linspace(.0, 1., num=21)
    f1_range = np.zeros_like(thresholds, dtype=float)
    precision_range = np.zeros_like(thresholds, dtype=float)
    recall_range = np.zeros_like(thresholds, dtype=float)
    for i, theta in enumerate(thresholds):
        predicted_id = (np.squeeze(p_y) >= theta).astype(int)
        f1_range[i] = f1_score(t_y, predicted_id)
        precision_range[i] = precision_score(t_y, predicted_id)
        recall_range[i] = recall_score(t_y, predicted_id)

    ax = axs[0, 1]
    ax.set_ylim([-.05, 1.05])
    ax.grid()
    lns1 = ax.plot(thresholds, precision_range, color='blue', label='Precision')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Precision')
    ax2 = ax.twinx()
    ax2.set_ylim([-.05, 1.05])
    ax2.set_ylabel('Recall')
    lns2 = ax2.plot(thresholds, recall_range, color='red', label='Recall')
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc='lower center')

    ax = axs[1, 1]
    ax.set_ylim([-.05, 1.05])
    ax.grid()
    ax.plot(thresholds, f1_range, color='blue')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1')


def main():
    np.random.seed(NP_SEED)
    tf.random.set_seed(TF_SEED)

    # classifier_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"  # @param {type:"string"}

    """
    data_root = tf.keras.utils.get_file(
      'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
       untar=True)
    """

    def load_dataset_1(file_name):
        file_names_df = pd.read_csv(file_name)

        ''' Map the class labels to strings '1' and '0' because scikit-learn train_test_split() requires it to split
        the dataset with stratification (stratification throws an exception if class labels are numbers). '''
        label_to_class = {'dandelion': '0', 'daisy': '1', 'roses': '0', 'sunflowers': '0', 'tulips': '0', 'extras': '0'}

        def map_it(file_name):
            pos = file_name.index('/')
            dir_name = file_name[:pos]
            the_class = label_to_class[dir_name]
            return the_class

        # Drop samples of the extended flowers
        # dataset if they are not wanted
        if not use_extended_dataset:
            entry_ids = file_names_df[file_names_df['file_name'].str.contains('extras/')].index
            file_names_df = file_names_df.drop(entry_ids)

        file_names_df['class'] = file_names_df['file_name'].map(map_it)

        return file_names_df

    def load_dataset_2(file_name):
        file_names_df = pd.read_csv(file_name)

        ''' Map the class labels to strings '1' and '0' because scikit-learn train_test_split() requires label classes
         to be strings, in order to split the dataset with stratification (stratification throws an exception if class 
         labels are numbers). '''

        file_names_df['class'] = file_names_df['multi_class'].map(lambda x: '1' if x == 51 else '0')
        file_names_df.drop(columns=['multi_class'], inplace=True)

        return file_names_df

    def load_augmented_dataset(file_name):
        file_names_df = pd.read_csv(file_name)

        ''' Map the class labels to strings '1' and '0' because scikit-learn train_test_split() requires label classes
         to be strings, in order to split the dataset with stratification (stratification throws an exception if class 
         labels are numbers). '''

        file_names_df.drop(columns=['label'], inplace=True)
        file_names_df['class'] = '1'

        return file_names_df

    file_names_df = load_dataset_2('flower_classes.csv')
    file_names_augmented_df = load_augmented_dataset('augmented.csv')
    file_names_df = pd.concat([file_names_df, file_names_augmented_df], axis='rows')
    file_names_df.reset_index(inplace=True, drop=True)

    # Randomly select and drop the given number of samples with positive classification
    if count_to_be_dropped > 0:
        idx_to_be_dropped = np.random.choice(file_names_df[file_names_df['class'] == '1'].index,
                                             size=count_to_be_dropped,
                                             replace=False)
        file_names_df = file_names_df.drop(idx_to_be_dropped)
        file_names_df.reset_index(inplace=True, drop=True)

    training_df, test_df = train_test_split(file_names_df, test_size=.15, stratify=file_names_df['class'])
    training_df, validation_df = train_test_split(training_df, test_size=.15/.85, stratify=training_df['class'])
    training_df.reset_index(inplace=True, drop=True)
    validation_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)

    ''' Now convert the class labels to integers, as required for the class_mode 'raw' by flow_from_dataframe().
    Using the class_mode 'raw' (and not 'binary') is a requirement for fit() to compute precision, recall and auc 
    metrics correctly; fit() doesn't compute them correctly if class_mode is set to 'binary'. '''
    training_df = training_df.astype({'class': 'int'})
    validation_df = validation_df.astype({'class': 'int'})
    test_df = test_df.astype({'class': 'int'})

    # data_root = '/home/fanta/.keras/datasets/flower_photos_binary'
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
    """,
   rotation_range=45,
   shear_range=.2,
   zoom_range=.3,
   validation_split=.2)"""

    # TODO try other interpolations, e.g. bicubic
    training_data = image_generator.flow_from_dataframe(dataframe=training_df,
                                                        directory=DATASET_ROOT,
                                                        x_col='file_name',
                                                        y_col='class',
                                                        target_size=IMAGE_SIZE,
                                                        class_mode='raw',
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=True)

    def make_validation_generator(shuffle=False):
        validation_data = image_generator.flow_from_dataframe(dataframe=validation_df,
                                                              directory=DATASET_ROOT,
                                                              x_col='file_name',
                                                              y_col='class',
                                                              target_size=IMAGE_SIZE,
                                                              class_mode='raw',
                                                              batch_size=VALIDATION_BATCH_SIZE,
                                                              shuffle=shuffle)

        return validation_data

    def make_test_generator():
        test_data = image_generator.flow_from_dataframe(dataframe=test_df,
                                                        directory=DATASET_ROOT,
                                                        x_col='file_name',
                                                        y_col='class',
                                                        target_size=IMAGE_SIZE,
                                                        class_mode='raw',
                                                        batch_size=VALIDATION_BATCH_SIZE,
                                                        shuffle=False)

        return test_data

    validation_data = make_validation_generator()

    visual_check_samples = make_validation_generator(shuffle=True)
    plot_classified_samples(visual_check_samples)
    plt.show()

    def make_model_MobileNetV2(output_bias=None):
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')
        for layer in base_model.layers:
            layer.trainable = False
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),  # TODO try with Flatten()
            tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
            # With linear activation, fit() won't be able to compute precision and recall
        ])
        return model

    def make_model_DenseNet121(output_bias=None):
        base_model = DenseNet121(include_top=False, pooling='avg', weights='imagenet', input_shape=IMAGE_SHAPE)
        for layer in base_model.layers:
            layer.trainable = False
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
            # With linear activation, fit() won't be able to compute precision and recall
        ])
        return model

    pos = np.sum(training_df['class'])
    assert sum(training_data.labels) == pos
    total = len(training_df)
    assert training_data.samples == total
    neg = total - pos
    print("Count of samples in training dataset: positive=", pos, ' negative=', neg, ' total=', total, sep='')

    model = make_model_DenseNet121(constant(np.log([pos / neg])))
    # model = make_model_MobileNetV2(constant(np.log([pos / neg])))
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),  # Consider others, e.g. RMSprop
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall(),
                 tf.keras.metrics.AUC()])

    '''Using restore_best_weights=False here because, even if set to True, it only works when early stopping actually 
    kicked in, see https://github.com/keras-team/keras/issues/12511
    To be sure to restore the best weights, will use the model saved by ModelCheckpoint() instead.'''
    early_stopping_CB = EarlyStopping(monitor='val_loss',
                                      # TODO this doesn't work correctly if there is L1 or L2 regularization
                                      mode='min',
                                      patience=10,
                                      restore_best_weights=False)

    checkpoint_CB = ModelCheckpoint(CHECKPOINTS_PATH,
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=False,
                                    save_weights_only=True)

    callbacks = [checkpoint_CB]

    steps_per_train_epoch = np.ceil(training_data.samples / training_data.batch_size)
    steps_per_val_epoch = np.ceil(validation_data.samples / validation_data.batch_size)

    # Compute weights for unbalanced dataset
    weight_for_0 = 2 * pos / total
    weight_for_1 = 2 * neg / total
    # Weights from Tensorflow tutorial
    # weight_for_0 = (1 / neg) * (total) / 2.0
    # weight_for_1 = (1 / pos) * (total) / 2.0
    # Equivalent to no weights
    # weight_for_0 = 1
    # weight_for_1 = 1
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print('Weight for class 0: {}'.format(weight_for_0))
    print('Weight for class 1: {}'.format(weight_for_1))

    for file in Path(CHECKPOINTS_DIR).glob('*.hdf5'):
        file.unlink()

    history = model.fit(training_data, epochs=EPOCHS,
                        steps_per_epoch=steps_per_train_epoch,
                        validation_data=validation_data,
                        validation_steps=steps_per_val_epoch,
                        class_weight=class_weight,
                        callbacks=callbacks,
                        verbose=1)

    plot_metrics(history)
    plt.show()

    # Before validation, reload the model with the best loss
    precision_val = np.array(history.history['val_precision'])
    recall_val = np.array(history.history['val_recall'])
    epsilon = 1e-7
    val_F1 = 2. * (precision_val * recall_val) / (precision_val + recall_val + epsilon)
    # Careful: if you change the choice of metric below, change min/max accordingly!
    # best_epoch = np.argmax(val_F1) + 1
    best_epoch = np.argmin(history.history['val_loss']) + 1
    if best_epoch == len(history.history['val_loss']):
        print('Best epoch is the last one, keeping it for validation')
    else:
        weights_file = CHECKPOINTS_PATH.format(epoch=best_epoch)
        print('Loading weights from file', weights_file, 'corresponding to best epoch', best_epoch)
        model.load_weights(weights_file)

    def print_metrics(samples, prediction):
        prediction = np.squeeze(prediction)
        assert len(prediction) == len(samples.labels)
        predicted_id = (np.squeeze(prediction) >= THETA).astype(int)
        print('   F1 score', f1_score(samples.labels, predicted_id))
        print('   Precision', precision_score(samples.labels, predicted_id))
        print('   Recall', recall_score(samples.labels, predicted_id))
        print('   Accuracy', accuracy_score(samples.labels, predicted_id))

    validation_data = make_validation_generator()
    validation_prediction = model.predict(validation_data, batch_size=validation_data.batch_size, verbose=1)
    print('Metrics on validation set:')
    print_metrics(validation_data, validation_prediction)

    test_data = make_test_generator()
    test_prediction = model.predict(test_data, batch_size=test_data.batch_size, verbose=1)
    print('Metrics on test set:')
    print_metrics(test_data, test_prediction)

    plot_auc_and_pr(test_data.labels, test_prediction)
    plt.show()

    validation_samples = make_validation_generator(shuffle=True)
    plot_misclassified_samples(validation_samples, model)
    plt.show()

    """ TODO
    Try different pre-trained models, also with fine-tuning (densenet)
    Introduce regularization, early stopping, lowering learning rate, resuming training
    try monitoring different metrics for early stopping, e.g. AUC or F1
    try validation with balanced classes
    Implement explainability
    on chest x-ray only: augment positive samples
    Framework to automate experiments
    """


if __name__ == '__main__':
    main()
