import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import numpy as np
import time
import PIL.Image as Image
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, \
    plot_precision_recall_curve, f1_score, confusion_matrix

IMAGE_SIZE = (224, 224)
IMG_SHAPE = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
EPOCHS = 5
BATCH_SIZE = 16
VALIDATION_BATCH_SIZE = 64
THETA = .5


def plot_history(history):
    epsilon = 1e-7
    epochs = [i + 1 for i in history.epoch]
    train_precision = np.array(history.history['precision'])
    train_recall = np.array(history.history['recall'])
    val_precision = np.array(history.history['val_precision'])
    val_recall = np.array(history.history['val_recall'])
    train_F1 = 2. * (train_precision * train_recall) / (
            train_precision + train_recall + epsilon)
    val_F1 = 2. * (val_precision * val_recall) / (val_precision + val_recall + epsilon)

    _, ax1 = plt.subplots()
    ax1.set_title('Training and Validation Metrics')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_xticks(epochs)
    ax1.plot(epochs, history.history["loss"], label="Train. loss", color='greenyellow')
    ax1.plot(epochs, history.history["val_loss"], label="Val. loss", color='darkolivegreen')

    ax2 = ax1.twinx()
    ax2.set_ylabel('F1')
    ax2.set_xticks(epochs)
    ax2.plot(epochs, train_F1, label="Train. F1", color='magenta')
    ax2.plot(epochs, val_F1, label="Val. F1", color='darkmagenta')

    ax1.legend(loc='center left')
    ax2.legend(loc='center right')


def plot_classified_samples(validation_samples, training_data, model, theta=THETA):
    class_names = sorted(training_data.class_indices.items(), key=lambda pair: pair[1])
    class_names = np.array([key.title() for key, value in class_names])

    for image_batch, label_batch in validation_samples:
        print("Image batch shape: ", image_batch.shape)
        print("Label batch shape: ", label_batch.shape)
        break
    predicted_batch = model.predict(image_batch)
    predicted_id = (np.squeeze(predicted_batch) >= theta).astype(int)
    predicted_label_batch = class_names[predicted_id]
    label_id = label_batch.astype(int)

    plt.figure(figsize=(10, 9))
    plt.subplots_adjust(hspace=0.5)
    for n in range(30):
        plt.subplot(6, 5, n + 1)
        plt.imshow(image_batch[n])
        color = "green" if predicted_id[n] == label_id[n] else "red"
        plt.title(predicted_label_batch[n].title(), color=color)
        plt.axis('off')
    _ = plt.suptitle("Model predictions (green: correct, red: incorrect)")


def plot_auc(t_y, p_y):
    fig, c_ax = plt.subplots(1, 1, figsize=(9, 9))
    fpr, tpr, thresholds = roc_curve(t_y, p_y, pos_label=1)
    c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % ('Dandelion', auc(fpr, tpr)))
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')


## what other performance statistics do you want to include here besides AUC?

def plot_pr(t_y, p_y):
    fig, c_ax = plt.subplots(1, 1, figsize=(9, 9))
    precision, recall, thresholds = precision_recall_curve(t_y, p_y)
    c_ax.plot(precision, recall, label='%s (AP Score:%0.2f)' % ('Dandelion', average_precision_score(t_y, p_y)))
    c_ax.legend()
    c_ax.set_xlabel('Recall')
    c_ax.set_ylabel('Precision')


def main():
    # classifier_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"  # @param {type:"string"}

    """
    data_root = tf.keras.utils.get_file(
      'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
       untar=True)
    """

    data_root = '/home/fanta/.keras/datasets/flower_photos_binary'
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255, validation_split=.2)
    # TODO try other interpolations, e.g. bicubic
    training_data = image_generator.flow_from_directory(str(data_root),
                                                        target_size=IMAGE_SIZE,
                                                        class_mode='binary',
                                                        classes=['non-daisy', 'daisy'],
                                                        subset='training',
                                                        batch_size=BATCH_SIZE)

    def make_validation_generator(shuffle=False):
        validation_data = image_generator.flow_from_directory(str(data_root),
                                                              target_size=IMAGE_SIZE,
                                                              class_mode='binary',
                                                              classes=['non-daisy', 'daisy'],
                                                              subset='validation',
                                                              batch_size=VALIDATION_BATCH_SIZE,
                                                              shuffle=shuffle)
        return validation_data

    validation_data = make_validation_generator()

    def make_model_MobileNetV2():
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')
        for layer in base_model.layers:
            layer.trainable = False
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()  # Try with Flatten()
        dense_layer = tf.keras.layers.Dense(320, activation='relu')
        prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        # prediction_layer = tf.keras.layers.Dense(1)
        model = tf.keras.Sequential([
            base_model,
            global_average_layer,
            # dense_layer,
            prediction_layer
        ])
        return model

    model = make_model_MobileNetV2()
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),  # Consider others, e.g. RMSprop
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall(),
                 tf.keras.metrics.AUC()])  # tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

    steps_per_train_epoch = np.ceil(training_data.samples / training_data.batch_size)
    steps_per_val_epoch = np.ceil(validation_data.samples / validation_data.batch_size)
    # batch_stats_callback = CollectBatchStats()

    history = model.fit(training_data, epochs=EPOCHS,
                        steps_per_epoch=steps_per_train_epoch,
                        validation_data=validation_data,
                        validation_steps=steps_per_val_epoch,
                        verbose=1)

    plot_history(history)
    plt.show()

    validation_samples = make_validation_generator(shuffle=True)

    plot_classified_samples(validation_samples, training_data, model)
    plt.show()

    validation_data = make_validation_generator()

    prediction = model.predict(validation_data, batch_size=validation_data.batch_size, verbose=1)
    prediction = np.squeeze(prediction)
    assert len(prediction) == len(validation_data.classes)
    predicted_id = (np.squeeze(prediction) >= THETA).astype(int)
    print('Accuracy on validation set', np.sum(validation_data.classes == predicted_id) / len(predicted_id))
    print('F1 score on validation set', f1_score(validation_data.classes, predicted_id))
    plot_auc(validation_data.classes, prediction)
    plt.show()
    plot_pr(validation_data.classes, prediction)
    plt.show()

    t = time.time()
    export_path = "/tmp/saved_models/{}".format(int(t))
    model.save(export_path, save_format='tf')
    # reloaded = tf.keras.models.load_model(export_path)

    """ TODO
    Add it under GitHub
    Make classes havily imbalanced, instroduce weighted loss
    Do it with few positive samples
    Introduce regularization, early stopping, lowering learning rate, resuming training
    Try different pre-trained models, also with fine-tuning
    """


if __name__ == '__main__':
    main()
