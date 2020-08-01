import matplotlib.pylab as plt
import random

# plt.ion()
print('Using', plt.get_backend(), 'as graphics backend.')
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from time import time
import PIL.Image as Image
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, \
    plot_precision_recall_curve, f1_score, confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.initializers import constant
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pathlib import Path
from tensorflow.keras.applications.densenet import DenseNet121


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


def plot_classified_samples(validation_samples):
    for image_batch, label_batch in validation_samples:
        print("Image batch shape: ", image_batch.shape)
        print("Label batch shape: ", label_batch.shape)
        break

    plt.figure(figsize=(10, 9))
    plt.subplots_adjust(hspace=0.5)
    for n in range(30):
        plt.subplot(6, 5, n + 1)
        plt.imshow(image_batch[n])
        plt.axis('off')
    _ = plt.suptitle("Model predictions (green: correct, red: incorrect)")


def plot_misclassified_samples(validation_samples, model, theta):
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
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
        # With linear activation, fit() won't be able to compute precision and recall
    ])
    return model


def run_experiment(params):
    n_epochs = params['n_epochs']
    batch_size = params['batch_size']
    val_batch_size = params['val_batch_size']
    test_set_fraction = params['test_set_fraction']
    augm_batch_size = params['augm_batch_size']
    augm_factor = params['augm_factor']
    theta = params['theta']
    count_to_be_dropped = params['count_to_be_dropped']
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

    np.random.seed(np_seed)
    tf.random.set_seed(tf_seed)
    random.seed(py_seed)

    # classifier_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"  # @param {type:"string"}

    """
    data_root = tf.keras.utils.get_file(
      'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
       untar=True)
    """

    file_names_df = load_dataset_('flower_classes.csv')

    # Randomly select and drop the given number of samples with positive classification
    if count_to_be_dropped > 0:
        idx_to_be_dropped = np.random.choice(file_names_df[file_names_df['class'] == '1'].index,
                                             size=count_to_be_dropped,
                                             replace=False)
        file_names_df = file_names_df.drop(idx_to_be_dropped)
        file_names_df.reset_index(inplace=True, drop=True)

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
    plot_classified_samples(visual_check_samples)
    plt.show()

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

    plot_metrics(history)
    plt.show()
    # plt.draw()
    # plt.pause(.01)

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

    def print_metrics(samples, prediction):
        prediction = np.squeeze(prediction)
        assert len(prediction) == len(samples.labels)
        predicted_id = (np.squeeze(prediction) >= theta).astype(int)
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
    # plt.draw()
    # plt.pause(.01)

    validation_samples = make_validation_generator(shuffle=True)
    plot_misclassified_samples(validation_samples, model, theta)
    plt.show()
    # plt.draw()
    # plt.pause(.01)

    # input('Press [Enter] to close charts and end the program.')


if __name__ == '__main__':
    params = {'n_epochs': 10,  # TODO use a named tuple instead?
              'batch_size': 24,
              'val_batch_size': 64,
              'test_set_fraction': .2,
              'augm_batch_size': 64,
              'augm_factor': 3,
              'theta': .5,
              'count_to_be_dropped': 0,
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

    run_experiment(params)

    """ 
    TODO
    Introduce tensorboard
    Check new memory profiles on tensorboard
    Introduce regularization, early stopping, lowering learning rate, resuming training
    try monitoring different metrics for early stopping, e.g. AUC or F1
    try validation with balanced classes
    Implement explainability
    Framework to automate experiments
    Make a dashboard
    """
