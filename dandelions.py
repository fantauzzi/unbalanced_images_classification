import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import numpy as np
import time
import PIL.Image as Image
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, \
    plot_precision_recall_curve, f1_score, confusion_matrix, precision_score, recall_score, accuracy_score

IMAGE_SIZE = (224, 224)
IMG_SHAPE = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
EPOCHS = 4
BATCH_SIZE = 16
VALIDATION_BATCH_SIZE = 64
THETA = .5


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


def plot_auc_and_pr(t_y, p_y):
    fig, (c_ax1, c_ax2) = plt.subplots(ncols=2, figsize=(8, 4))

    fpr, tpr, thresholds = roc_curve(t_y, p_y, pos_label=1)
    c_ax1.grid()
    c_ax1.plot(fpr, tpr, label='%s (AUC:%0.2f)' % ('Dandelion', auc(fpr, tpr)))
    c_ax1.legend()
    c_ax1.set_xlabel('False Positive Rate')
    c_ax1.set_ylabel('True Positive Rate')

    precision, recall, thresholds = precision_recall_curve(t_y, p_y)
    c_ax2.grid()
    c_ax2.plot(precision, recall, label='%s (AP Score:%0.2f)' % ('Dandelion', average_precision_score(t_y, p_y)))
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
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),  # TODO try with Flatten()
            # tf.keras.layers.Dense(320, activation='relu')
            tf.keras.layers.Dense(1, activation='sigmoid')
            # With linear activation, fit() won't be able to compute precision and recall
        ])
        return model

    model = make_model_MobileNetV2()
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),  # Consider others, e.g. RMSprop
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall(),
                 tf.keras.metrics.AUC()])

    steps_per_train_epoch = np.ceil(training_data.samples / training_data.batch_size)
    steps_per_val_epoch = np.ceil(validation_data.samples / validation_data.batch_size)

    # Compute weights for unbalanced dataset
    pos = np.sum(training_data.classes)
    total = len(training_data.classes)
    neg = total - pos
    # weight_for_0 = (1 / neg) * (total) / 2.0
    # weight_for_1 = (1 / pos) * (total) / 2.0
    weight_for_0 = 2 * pos / total
    weight_for_1 = 2 * neg / total
    # weight_for_0 = .5
    # weight_for_1 = .5
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print('Weight for class 0: {}'.format(weight_for_0))
    print('Weight for class 1: {}'.format(weight_for_1))

    history = model.fit(training_data, epochs=EPOCHS,
                        steps_per_epoch=steps_per_train_epoch,
                        validation_data=validation_data,
                        validation_steps=steps_per_val_epoch,
                        # class_weight=class_weight,
                        verbose=1)

    plot_metrics(history)
    plt.show()

    validation_samples = make_validation_generator(shuffle=True)

    # plot_classified_samples(validation_samples, training_data, model)
    # plt.show()

    validation_data = make_validation_generator()

    prediction = model.predict(validation_data, batch_size=validation_data.batch_size, verbose=1)
    prediction = np.squeeze(prediction)
    assert len(prediction) == len(validation_data.classes)
    predicted_id = (np.squeeze(prediction) >= THETA).astype(int)
    print('Metrics on validation set:')
    print('   F1 score', f1_score(validation_data.classes, predicted_id))
    print('   Precision', precision_score(validation_data.classes, predicted_id))
    print('   Recall', recall_score(validation_data.classes, predicted_id))
    print('   Accuracy', accuracy_score(validation_data.classes, predicted_id))
    plot_auc_and_pr(validation_data.classes, prediction)
    plt.show()

    t = time.time()
    export_path = "/tmp/saved_models/{}".format(int(t))
    model.save(export_path, save_format='tf')
    # reloaded = tf.keras.models.load_model(export_path)

    """ TODO
    Make classes heavily imbalanced, introduce weighted loss
    Do it with few positive samples
    Introduce regularization, early stopping, lowering learning rate, resuming training
    Try different pre-trained models, also with fine-tuning
    Introduce  bias_initializer=output_bias and check if it helps
    try monitoring different metrics for early stopping, e.g. AUC or F1
    try validation with balanced classes
    """


if __name__ == '__main__':
    main()
