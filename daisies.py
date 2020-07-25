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

IMAGE_SIZE = (224, 224)
IMG_SHAPE = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
EPOCHS = 20
BATCH_SIZE = 16
VALIDATION_BATCH_SIZE = 64
THETA = .5
DATASET_ROOT = '/home/fanta/.keras/datasets/flower_photos'
CHECKPOINTS_DIR = 'checkpoints'
CHECKPOINTS_PATH = CHECKPOINTS_DIR + '/weights.{epoch:05d}.hdf5'
count_to_be_dropped = 333


def plot_metrics(history):
    epsilon = 1e-7
    epochs = [i + 1 for i in history.epoch]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    metrics = ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(3, 2, n + 1)
        plt.grid(True)
        plt.plot(epochs, history.history[metric], color=colors[0], linestyle="--", label='Train')
        plt.plot(epochs, history.history['val_' + metric],
                 color=colors[0], label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([min(min(history.history['loss']), min(history.history['val_loss'])),
                      max(max(history.history['loss']), max(history.history['val_loss']))])
        elif metric == 'auc':
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])
        plt.xticks(epochs)
        plt.legend()
    plt.subplot(3, 2, 5)
    precision = np.array(history.history['precision'])
    recall = np.array(history.history['recall'])
    precision_val = np.array(history.history['val_precision'])
    recall_val = np.array(history.history['val_recall'])
    train_F1 = 2. * (precision * recall) / (precision + recall + epsilon)
    val_F1 = 2. * (precision_val * recall_val) / (precision_val + recall_val + epsilon)
    plt.grid(True)
    plt.plot(epochs, train_F1, color=colors[0], linestyle="--", label='Train')
    plt.plot(epochs, val_F1, color=colors[0], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.ylim([0, 1])
    plt.xticks(epochs)
    plt.legend()
    plt.subplots_adjust(wspace=.3, hspace=.3)


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


def plot_auc_and_pr(t_y, p_y):
    fig, (c_ax1, c_ax2) = plt.subplots(ncols=2, figsize=(8, 4))

    fpr, tpr, thresholds = roc_curve(t_y, p_y, pos_label=1)
    c_ax1.grid()
    c_ax1.plot(fpr, tpr, label='%s (AUC:%0.2f)' % ('Daisies', auc(fpr, tpr)))
    c_ax1.legend()
    c_ax1.set_xlabel('False Positive Rate')
    c_ax1.set_ylabel('True Positive Rate')

    precision, recall, thresholds = precision_recall_curve(t_y, p_y)
    c_ax2.grid()
    c_ax2.plot(precision, recall, label='%s (AP Score:%0.2f)' % ('Daisies', average_precision_score(t_y, p_y)))
    c_ax2.legend()
    c_ax2.set_xlabel('Recall')
    c_ax2.set_ylabel('Precision')


def main():
    # classifier_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"  # @param {type:"string"}

    """
    data_root = tf.keras.utils.get_file(
      'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
       untar=True)
    """
    file_names_df = pd.read_csv('file_names.txt')

    ''' Map the class labels to strings '1' and '0' because scikit-learn train_test_split() requires it to split
    the dataset with stratification (stratification throws an exception if class labels are numbers). '''
    label_to_class = {'dandelion': '0', 'daisy': '1', 'roses': '0', 'sunflowers': '0', 'tulips': '0'}

    def map_it(file_name):
        pos = file_name.index('/')
        dir_name = file_name[:pos]
        the_class = label_to_class[dir_name]
        return the_class

    file_names_df['class'] = file_names_df['file_name'].map(map_it)

    # Randomly select and drop the given number of samples with positive classification
    if count_to_be_dropped > 0:
        np.random.seed(41)
        idx_to_be_dropped = np.random.choice(file_names_df[file_names_df['class'] == '1'].index,
                                             size=count_to_be_dropped,
                                             replace=False)
        file_names_df = file_names_df.drop(idx_to_be_dropped)
        file_names_df.reset_index(inplace=True, drop=True)

    training_df, validation_df = train_test_split(file_names_df, test_size=.2, stratify=file_names_df['class'])
    training_df.reset_index(inplace=True, drop=True)
    validation_df.reset_index(inplace=True, drop=True)

    ''' Now convert the class labels to integers, as required for the class_mode 'raw' by flow_from_dataframe().
    Using the class_mode 'raw' (and not 'binary') is a requirement for fit() to compute precision, recall and auc 
    metrics correctly; fit() doesn't compute them correctly if class_mode is set to 'binary'. '''
    training_df = training_df.astype({'class': 'int'})
    validation_df = validation_df.astype({'class': 'int'})

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

    validation_data = make_validation_generator()

    visual_check_samples = make_validation_generator(shuffle=True)
    plot_classified_samples(visual_check_samples)
    plt.show()

    def make_model_MobileNetV2(output_bias=None):
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
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

    pos = np.sum(training_df['class'])
    assert sum(training_data.labels) == pos
    total = len(training_df)
    assert training_data.samples == total
    neg = total - pos
    print("Count of samples in training dataset: positive=", pos, ' negative=', neg, ' total=', total, sep='')

    model = make_model_MobileNetV2(constant(np.log([pos / neg])))
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
    # weight_for_0 = (1 / neg) * (total) / 2.0
    # weight_for_1 = (1 / pos) * (total) / 2.0
    weight_for_0 = 2 * pos / total
    weight_for_1 = 2 * neg / total
    # weight_for_0 = .5
    # weight_for_1 = .5
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
    best_epoch = np.argmax(val_F1) + 1  # Careful: if you change the metric, change min/max accordingly!
    if best_epoch == len(history.history['val_loss']):
        print('Best epoch is the last one, keeping it for validation')
    else:
        weights_file = CHECKPOINTS_PATH.format(epoch=best_epoch)
        print('Loading weights from file', weights_file, 'corresponding to best epoch', best_epoch)
        model.load_weights(weights_file)

    validation_data = make_validation_generator()

    prediction = model.predict(validation_data, batch_size=validation_data.batch_size, verbose=1)
    prediction = np.squeeze(prediction)
    assert len(prediction) == len(validation_data.labels)
    predicted_id = (np.squeeze(prediction) >= THETA).astype(int)
    print('Metrics on validation set:')
    print('   F1 score', f1_score(validation_data.labels, predicted_id))
    print('   Precision', precision_score(validation_data.labels, predicted_id))
    print('   Recall', recall_score(validation_data.labels, predicted_id))
    print('   Accuracy', accuracy_score(validation_data.labels, predicted_id))
    plot_auc_and_pr(validation_data.labels, prediction)
    plt.show()

    validation_samples = make_validation_generator(shuffle=True)
    plot_classified_samples(validation_samples, model)
    plt.show()

    t = time.time()
    export_path = "/tmp/saved_models/{}".format(int(t))
    model.save(export_path, save_format='tf')
    # reloaded = tf.keras.models.load_model(export_path)

    """ TODO
    Make classes heavily imbalanced
    Split chart for loss in two (tain and val)
    Introduce regularization, early stopping, lowering learning rate, resuming training
    try monitoring different metrics for early stopping, e.g. AUC or F1
    Try different pre-trained models, also with fine-tuning
    try validation with balanced classes
    """


if __name__ == '__main__':
    main()
