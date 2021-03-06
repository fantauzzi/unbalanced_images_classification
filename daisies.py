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
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
import csv
from datetime import datetime
import pickle


def plot_samples(samples, fig_no):
    for image_batch, label_batch in samples:
        break

    the_figure = plt.figure(num=fig_no, figsize=(10, 9))
    the_figure.clear()
    plt.subplots_adjust(hspace=0.5)
    for n in range(30):
        plt.subplot(6, 5, n + 1)
        plt.imshow(image_batch[n])
        plt.title(label_batch[n])
        plt.axis('off')
    _ = plt.suptitle("Samples from the dataset (including any augmentation)")

    plt.draw()
    plt.pause(.01)


def plot_misclassified_samples(validation_samples, model, theta, fig_no):
    max_to_be_plotted = 30
    to_be_plotted = []
    to_be_plotted_label = []
    # Find the first max_to_be_plotted images that are misclassified by the model
    steps = np.ceil(validation_samples.samples / validation_samples.batch_size)
    for image_batch, label_batch in validation_samples:
        if validation_samples.total_batches_seen > steps:
            break
        predicted_batch = model.predict(image_batch)
        ''' Don't use np.squeeze() here below, because otherwise an image_batch of size 1 becomes a scalar (instead
        of the wanted array of lenght 1) '''
        predicted_batch_id = (predicted_batch[:, 0] >= theta).astype(int)
        label_batch_id = label_batch.astype(int)
        correct_label_batch = np.array(['Positive' if item == 1 else 'Negative' for item in label_batch_id])
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

    the_figure = plt.figure(num=fig_no, figsize=(10, 9))
    the_figure.clear()
    plt.subplots_adjust(hspace=0.5)
    for n in range(len(to_be_plotted)):
        plt.subplot(6, 5, n + 1)
        plt.imshow(to_be_plotted[n])
        plt.title(to_be_plotted_label[n])
        plt.axis('off')
    _ = plt.suptitle("Sample of misclassified images with their correct classification")

    plt.draw()
    plt.pause(.01)


def display_dashboard(history, t_y, p_y):
    def format_axes(fig):
        for i, ax in enumerate(fig.axes):
            ax.text(0.5, 0.5, "ax%d" % (i + 1), va="center", ha="center")
            ax.tick_params(labelbottom=False, labelleft=False)

    epsilon = 1e-7
    epochs = [i + 1 for i in history.epoch]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig = plt.figure(num=1, figsize=(18, 9), constrained_layout=True)
    fig.clear()

    gs = GridSpec(nrows=6, ncols=12, figure=fig)

    ax_loss = fig.add_subplot(gs[0:2, 0:3])
    ax_loss.set_title('Loss')
    ax_loss.grid(True, axis='x')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Training Loss')
    ax_loss2 = ax_loss.twinx()
    ax_loss2.set_ylabel('Validation Loss')
    lns1 = ax_loss.plot(epochs, history.history['loss'], color=colors[0], linestyle="--",
                        label='Train')
    lns2 = ax_loss2.plot(epochs, history.history['val_loss'], color=colors[0], linestyle="-",
                         label='Validation')
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax_loss2.legend(lns, labs, loc=0)

    ax_auc = fig.add_subplot(gs[0:2, 3:6])
    ax_auc.set_title('AUC')
    ax_auc.set_xlabel('Epoch')
    ax_auc.set_ylabel('AUC')
    ax_auc.plot(epochs, history.history['auc'], color=colors[0], linestyle="--",
                label='Train')
    ax_auc.plot(epochs, history.history['val_auc'], color=colors[0], linestyle="-",
                label='Validation')

    precision = np.array(history.history['precision'])
    precision_val = np.array(history.history['val_precision'])
    ax_pre = fig.add_subplot(gs[2:4, 0:3])
    ax_pre.set_title('Precision')
    ax_pre.set_xlabel('Epoch')
    ax_pre.set_ylabel('Precision')
    ax_pre.plot(epochs, precision, color=colors[0], linestyle="--",
                label='Train')
    ax_pre.plot(epochs, precision_val, color=colors[0], linestyle="-",
                label='Validation')

    recall = np.array(history.history['recall'])
    recall_val = np.array(history.history['val_recall'])
    ax_recall = fig.add_subplot(gs[2:4, 3:6])
    ax_recall.set_title('Recall')
    ax_recall.set_xlabel('Epoch')
    ax_recall.set_ylabel('Recall')
    ax_recall.plot(epochs, recall, color=colors[0], linestyle="--",
                   label='Train')
    ax_recall.plot(epochs, recall_val, color=colors[0], linestyle="-",
                   label='Validation')

    train_F1 = 2. * (precision * recall) / (precision + recall + epsilon)
    val_F1 = 2. * (precision_val * recall_val) / (precision_val + recall_val + epsilon)
    ax_f1 = fig.add_subplot(gs[4:6, 0:3])
    ax_f1.set_title('F1')
    ax_f1.set_xlabel('Epoch')
    ax_f1.set_ylabel('F1')
    ax_f1.plot(epochs, train_F1, color=colors[0], linestyle="--",
               label='Train')
    ax_f1.plot(epochs, val_F1, color=colors[0], linestyle="-",
               label='Validation')

    fpr, tpr, _ = roc_curve(t_y, p_y, pos_label=1)
    ax_roc = fig.add_subplot(gs[0:3, 6:9])
    ax_roc.set_title('ROC')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.plot(fpr, tpr, label='%s (AUC:%0.2f)' % ('Positives', auc(fpr, tpr)))

    thresholds = np.linspace(.0, 1., num=21)
    f1_range = np.zeros_like(thresholds, dtype=float)
    precision_range = np.zeros_like(thresholds, dtype=float)
    recall_range = np.zeros_like(thresholds, dtype=float)
    for i, theta in enumerate(thresholds):
        predicted_id = (p_y >= theta).astype(int)
        f1_range[i] = f1_score(t_y, predicted_id)
        precision_range[i] = precision_score(t_y, predicted_id)
        recall_range[i] = recall_score(t_y, predicted_id)

    ax_prt = fig.add_subplot(gs[0:3, 9:12])
    ax_prt.set_title('Precision and Recall to Threshold')
    ax_prt.set_xlabel('Threshold')
    ax_prt.set_ylabel('Precision')
    ax_prt2 = ax_prt.twinx()
    ax_prt2.set_ylabel('Recall')
    lns1 = ax_prt.plot(thresholds, precision_range, color='blue', label='Precision')
    lns2 = ax_prt2.plot(thresholds, recall_range, color='red', label='Recall')
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax_prt2.legend(lns, labs, loc='lower center')

    test_precision, test_recall, _ = precision_recall_curve(t_y, p_y)
    ax_pr = fig.add_subplot(gs[3:6, 6:9])
    ax_pr.set_title('Precision-Recall')
    ax_pr.set_xlabel('Precision')
    ax_pr.set_ylabel('Recall')
    ax_pr.plot(test_precision, test_recall,
               label='%s (AP Score:%0.2f)' % ('Positives', average_precision_score(t_y, p_y)))

    ax_f1t = fig.add_subplot(gs[3:6, 9:12])
    ax_f1t.set_title('F1 to Threshold')
    ax_f1t.set_xlabel('Threshold')
    ax_f1t.set_ylabel('F1')
    ax_f1t.plot(thresholds, f1_range, color='blue')

    fig.suptitle('Dashboard')
    # format_axes(fig)

    for ax in fig.get_axes():
        if ax.get_title() not in ('Loss', '', 'Precision and Recall to Threshold', 'F1 to Threashold'):
            ax.legend()
        if ax.get_title() not in ('Loss', ''):
            ax.grid(True)

    plt.draw()
    plt.pause(.01)


def augment_positive_samples(file_names_df, output_file_name, params):
    file_names_df = file_names_df[file_names_df['class'] == '1']
    # print('Loaded information for', len(file_names_df), 'files with positive samples.')
    print('Making', params['augm_factor'], 'new samples for every positive sample.')

    for file in Path(params['augm_target_dir']).glob('*.png'):
        file.unlink()

    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=90,
                                                                      shear_range=.2,
                                                                      zoom_range=(.75, 1.20),
                                                                      brightness_range=(.7, 1.2))

    # TODO try other interpolations, e.g. bicubic
    generated_data = image_generator.flow_from_dataframe(dataframe=file_names_df,
                                                         directory=params['dataset_root'],
                                                         save_to_dir=params['augm_target_dir'],
                                                         target_size=params['image_size'],
                                                         x_col='file_name',
                                                         y_col='class',
                                                         class_mode='raw',
                                                         batch_size=params['augm_batch_size'],
                                                         shuffle=False)
    wanted_batches = int(params['augm_factor'] * np.ceil(generated_data.samples / generated_data.batch_size))
    generated_images_count = 0
    for image_batch, label_batch in generated_data:
        assert sum(label_batch == '1') == len(image_batch)
        generated_images_count += len(image_batch)
        print('.', end='', flush=True)
        if generated_data.total_batches_seen == wanted_batches:
            break
    print('\nGenerated', generated_images_count, 'images in', params['augm_target_dir'])

    new_rows = []
    with open(output_file_name, 'w') as target_file:
        target_file.write('file_name,label,class\n')
        for file_name in Path(params['augm_target_dir']).glob('*.png'):
            to_be_written = params['augm_target_subdir'] + '/' + str(file_name.name)
            target_file.write(to_be_written + ',51\n')
            new_rows.append({'file_name': to_be_written, 'class': '1'})
    print('Written metadata file', params['aug_metadata_file_name'])
    metadata_df = pd.DataFrame(new_rows)
    return metadata_df


def load_dataset_(file_name):
    file_names_df = pd.read_csv(file_name)

    ''' Map the class labels to strings '1' and '0' because scikit-learn train_test_split() requires label classes
     to be strings, in order to split the dataset with stratification (stratification throws an exception if class 
     labels are numbers). '''

    file_names_df['class'] = file_names_df['multi_class'].map(lambda x: '1' if x == 51 else '0')
    file_names_df.drop(columns=['multi_class'], inplace=True)

    return file_names_df


def make_model_MobileNetV2(image_shape, output_bias=None):
    base_model = tf.keras.applications.MobileNetV2(input_shape=image_shape,
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


def make_model_DenseNet121(image_shape, output_bias=None):
    base_model = DenseNet121(include_top=False, pooling='avg', weights='imagenet', input_shape=image_shape)
    # for layer in base_model.layers:
    for layer in base_model.layers[0:313]:
        layer.trainable = False
    model = tf.keras.Sequential([
        base_model,
        # tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
        # With linear activation, fit() won't be able to compute precision and recall
    ])
    return model


def run_experiment(params):
    batch_size = int(params['batch_size'])
    augm_factor = params['augm_factor']
    test_set_fraction = params['test_set_fraction']
    n_epochs = params['n_epochs']
    val_batch_size = params['val_batch_size']
    augm_batch_size = params['augm_batch_size']
    theta = params['theta']
    image_size = params['image_size']
    image_shape = params['image_shape']
    dataset_root = params['dataset_root']
    checkpoints_dir = params['checkpoints_dir']
    checkpoints_path = params['checkpoints_path']
    augm_target_subdir = params['augm_target_subdir']
    augm_target_dir = params['augm_target_dir']
    aug_metadata_file_name = params['aug_metadata_file_name']
    val_results_file_name = params['val_results_file_name']
    py_seed = params['py_seed']
    np_seed = params['np_seed']
    tf_seed = params['tf_seed']

    np.random.seed(np_seed)
    tf.random.set_seed(tf_seed)
    random.seed(py_seed)

    samples_fig_no = 2

    # classifier_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"  # @param {type:"string"}

    """
    data_root = tf.keras.utils.get_file(
      'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
       untar=True)
    """

    file_names_df = load_dataset_('flower_classes.csv')

    training_df, test_df = train_test_split(file_names_df, test_size=test_set_fraction, stratify=file_names_df['class'])
    training_df, validation_df = train_test_split(training_df, test_size=test_set_fraction / (1 - test_set_fraction),
                                                  stratify=training_df['class'])
    if augm_factor != 0:
        file_names_augmented_df = augment_positive_samples(file_names_df=training_df,
                                                           output_file_name=aug_metadata_file_name,
                                                           params=params)
        training_df = pd.concat([training_df, file_names_augmented_df], axis='rows')

    # Shuffle the training dataset, shouldn't be necessary as the generator does it as well, but cannot find out
    # for sure in the documentation if the generato shuffles withing the batch or the whole dataset
    training_df = training_df.sample(frac=1, replace=False)
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

    # TODO try other interpolations, e.g. bicubic
    training_data = image_generator.flow_from_dataframe(dataframe=training_df,
                                                        directory=dataset_root,
                                                        x_col='file_name',
                                                        y_col='class',
                                                        target_size=image_size,
                                                        class_mode='raw',
                                                        batch_size=batch_size,
                                                        shuffle=True)

    def make_validation_generator(shuffle=False):
        validation_data = image_generator.flow_from_dataframe(dataframe=validation_df,
                                                              directory=dataset_root,
                                                              x_col='file_name',
                                                              y_col='class',
                                                              target_size=image_size,
                                                              class_mode='raw',
                                                              batch_size=val_batch_size,
                                                              shuffle=shuffle)

        return validation_data

    def make_test_generator():
        test_data = image_generator.flow_from_dataframe(dataframe=test_df,
                                                        directory=dataset_root,
                                                        x_col='file_name',
                                                        y_col='class',
                                                        target_size=image_size,
                                                        class_mode='raw',
                                                        batch_size=val_batch_size,
                                                        shuffle=False)

        return test_data

    validation_data = make_validation_generator()

    visual_check_samples = make_validation_generator(shuffle=True)
    plot_samples(visual_check_samples, fig_no=samples_fig_no)
    # plt.show()

    # plt.draw()
    # plt.pause(.01)

    pos = np.sum(training_df['class'])
    assert sum(training_data.labels) == pos
    total = len(training_df)
    assert training_data.samples == total
    neg = total - pos
    print("Count of samples in training dataset: positive=", pos, ' negative=', neg, ' total=', total, sep='')

    model = make_model_DenseNet121(image_shape, constant(np.log([pos / neg])))
    # model = make_model_MobileNetV2(image_shape, constant(np.log([pos / neg])))
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),  # Consider others, e.g. RMSprop
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'), tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')])

    '''Using restore_best_weights=False here because, even if set to True, it only works when early stopping actually 
    kicked in, see https://github.com/keras-team/keras/issues/12511
    To be sure to restore the best weights, will use the model saved by ModelCheckpoint() instead.'''
    early_stopping_CB = EarlyStopping(monitor='val_loss',
                                      # TODO this doesn't work correctly if there is L1 or L2 regularization
                                      mode='min',
                                      patience=10,
                                      restore_best_weights=False)

    checkpoint_CB = ModelCheckpoint(checkpoints_path,
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

    for file in Path(checkpoints_dir).glob('*.hdf5'):
        file.unlink()

    start_time = time()

    history = model.fit(training_data, epochs=n_epochs,
                        steps_per_epoch=steps_per_train_epoch,
                        validation_data=validation_data,
                        validation_steps=steps_per_val_epoch,
                        class_weight=class_weight,
                        callbacks=callbacks,
                        verbose=1)

    end_time = time()
    elapsed = int(end_time - start_time)
    print("Completed training in {}'{}''.".format(elapsed // 60, elapsed % 60))

    # Before validation, reload the model with the best loss
    # Careful: if you change the choice of metric below, change min/max accordingly!
    # best_epoch = np.argmax(val_F1) + 1
    best_epoch = np.argmin(history.history['val_loss']) + 1
    if best_epoch == len(history.history['val_loss']):
        print('Best epoch is the last one, keeping it for validation')
    else:
        weights_file = checkpoints_path.format(epoch=best_epoch)
        print('Loading weights from file', weights_file, 'corresponding to best epoch', best_epoch)
        model.load_weights(weights_file)

    def compute_metrics(samples, prediction):
        prediction = prediction[:, 0]
        assert len(prediction) == len(samples.labels)
        # predicted_id = (prediction[:, 0] >= theta).astype(int)
        predicted_id = (prediction >= theta).astype(int)
        f1 = f1_score(samples.labels, predicted_id)
        pre = precision_score(samples.labels, predicted_id)
        recall = recall_score(samples.labels, predicted_id)
        accu = accuracy_score(samples.labels, predicted_id)
        return f1, pre, recall, accu

    def print_metrics(f1, pre, recall, accu):
        print('   F1 score', f1)
        print('   Precision', pre)
        print('   Recall', recall)
        print('   Accuracy', accu)

    validation_data = make_validation_generator()
    validation_prediction = model.predict(validation_data, batch_size=validation_data.batch_size, verbose=1)
    print('Metrics on validation set:')
    val_metrics = compute_metrics(validation_data, validation_prediction)
    print_metrics(*val_metrics)
    val_loss = history.history['val_loss'][best_epoch - 1]
    fpr, tpr, _ = roc_curve(validation_data.labels, validation_prediction, pos_label=1)
    val_auc = auc(fpr, tpr)
    val_ap = average_precision_score(validation_data.labels, validation_prediction)

    with open(val_results_file_name, 'a') as results_file:
        writer = csv.writer(results_file)
        writer.writerow(
            [str(datetime.now()), elapsed] + [params[param_name] for param_name in sorted(params.keys())] + list(
                val_metrics[:3]) + [val_auc, val_ap,
                                    val_loss])

    test_data = make_test_generator()
    test_prediction = model.predict(test_data, batch_size=test_data.batch_size, verbose=1)
    test_prediction = test_prediction[:, 0]
    print('Metrics on test set:')
    test_metrics = compute_metrics(validation_data, validation_prediction)
    print_metrics(*test_metrics)

    display_dashboard(history, test_data.labels, test_prediction)

    validation_samples = make_validation_generator(shuffle=True)
    plot_misclassified_samples(validation_samples, model, theta, fig_no=samples_fig_no)

    return val_loss


if __name__ == '__main__':
    params = {'n_epochs': 3,  # TODO use a named tuple instead?
              'batch_size': hp.quniform('batch_size', 16, 32, 1),
              'val_batch_size': 64,
              'test_set_fraction': hp.uniform('test_set_fraction', .20, .30),
              'augm_batch_size': 64,
              'augm_factor': hp.choice('augm_factor', (1, 2, 3, 4)),
              'theta': .5,
              'image_shape': (224, 224, 3),
              'dataset_root': '/home/fanta/.keras/datasets/102flowers/jpg',
              'checkpoints_dir': 'checkpoints',
              'augm_target_subdir': 'augmented',
              'aug_metadata_file_name': 'augmented.csv',
              'val_results_file_name': 'validation_results.csv',
              'py_seed': 44,
              'np_seed': 43,
              'tf_seed': 42}

    params['checkpoints_path'] = params['checkpoints_dir'] + '/weights.{epoch:05d}.hdf5'
    params['augm_target_dir'] = params['dataset_root'] + '/' + params['augm_target_subdir']
    params['image_size'] = params['image_shape'][:2]

    with open(params['val_results_file_name'], 'a') as results_file:
        writer = csv.writer(results_file)
        writer.writerow(
            ['Timestamp', 'Elapsed (sec)'] + [param_name for param_name in sorted(params.keys())] + ['F1', 'Precision',
                                                                                                     'Recall',
                                                                                                     'ROC AUC',
                                                                                                     'AP',
                                                                                                     'Loss'])
    tpe_algorithm = tpe.suggest
    trials_path = Path('trials.pickle')
    max_iter = 10

    if trials_path.exists():
        with open(trials_path, 'rb') as pickle_file:
            bayes_trials = pickle.load(pickle_file)
        print('Loaded status of hyper-parameters tuning from file {}, resuming since iteration {}.'.format(
            trials_path,
            len(
                bayes_trials) + 1))
    else:
        bayes_trials = Trials()

    for i in range(max_iter):
        print('Trial no.', len(bayes_trials) + 1)
        best = fmin(fn=run_experiment,
                    space=params,
                    algo=tpe.suggest,
                    max_evals=len(bayes_trials) + 1,
                    trials=bayes_trials,
                    show_progressbar=False)
        with open(trials_path, 'wb') as pickle_file:
            pickle.dump(bayes_trials, pickle_file, pickle.HIGHEST_PROTOCOL)

    print(bayes_trials.best_trial)
    input('All done. Press [Enter] to end.')

    """ 
    TODO
    insert dataset with x-rays
    explainability
    Introduce tensorboard
    Check new memory profiles on tensorboard
    Introduce regularization, early stopping, lowering learning rate, resuming training
    try monitoring different metrics for early stopping, e .g. AUC or F1
    try validation with balanced classes
    Implement explainability
    """
