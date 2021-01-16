#!/usr/bin/env python
# coding: utf-8

# Imports
import gpflow
from gpflow.utilities import print_summary, set_trainable
from gpflow.ci_utils import ci_niter
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, classification_report
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import callbacks
import time


# This ensures that all the data isn't loaded into the GPU memory at once.
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)


def load_dataset(value):
    """This function loads the preprocessed dataset and the class mapping for the preprocessed dataset.

    :param value: Integer specifying the number of objects per class.
    :return dataset: The preprocessed dataset.
    :return class_map: The class mapping for the preprocessed dataset."""

    dataset = pd.read_csv("./preprocessed_datasets/preprocessed_dataset_{}.csv".format(value))
    print('Dataset Shape:', dataset.shape)
    
    class_names = pd.read_csv("./preprocessed_datasets/class_mapping_{}.csv".format(value), header=None)
    class_map = dict(class_names.values[:, ::-1])
    
    return dataset, class_map


def get_data(dataset):
    """This function creates the test and train datasets.

    :param dataset: The dataset from which we want to create the test and train datasets.
    :return x_train: The train dataset.
    :return y_train: The train dataset labels.
    :return x_test: The test dataset.
    :return y_test: The test dataset labels."""

    x = np.array(dataset.iloc[:, 1:-1])
    y = np.array(dataset.iloc[:, -1]).reshape(-1,)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    scaler = StandardScaler()
    
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    print('X_train: ', x_train.shape, ', y_train: ', y_train.shape, ', X_test: ', x_test.shape, ', y_test: ',
          y_test.shape)

    return x_train, y_train, x_test, y_test


def gp_model(x_train, y_train, x_test, num_classes):
    """This function instantiates the gp model and gets the predictions from the model.

    :param x_train: The training dataset.
    :param y_train: The training dataset labels.
    :param x_test: The test dataset.
    :param num_classes: The number of classes in the dataset.
    :return: predictions, the predictions from the gp model.
    :return time_taken: The time taken to train the model."""

    data = (x_train, y_train)
    kernel = gpflow.kernels.SquaredExponential() + gpflow.kernels.Matern12() + gpflow.kernels.Exponential()

    invlink = gpflow.likelihoods.RobustMax(num_classes)
    likelihood = gpflow.likelihoods.MultiClass(num_classes, invlink=invlink)
    z = x_train[::5].copy()

    model = gpflow.models.SVGP(kernel=kernel, likelihood=likelihood, inducing_variable=z, num_latent_gps=num_classes,
                               whiten=True, q_diag=True)

    set_trainable(model.inducing_variable, False)
    
    print('\nInitial parameters:')
    print_summary(model, fmt="notebook")
    
    start = time.time()
    
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss_closure(data), model.trainable_variables, options=dict(maxiter=ci_niter(1000)))
    
    print('\nParameters after optimization:')
    print_summary(model, fmt="notebook")
    
    end = time.time()
    time_taken = round(end - start, 2)
    
    print('Optimization took {:.2f} seconds'.format(time_taken))

    predictions = model.predict_y(x_test)[0]

    return predictions, time_taken


def dl_model_1(x_train, y_train, x_test, y_test, num_classes):
    """This function instantiates the dl model and gets the predictions from the model.

    :param x_train: The training dataset.
    :param y_train: The training dataset labels.
    :param x_test: The test dataset.
    :param y_test: The test dataset labels.
    :param num_classes: The number of classes in the dataset.
    :return: predictions, the predictions from the gp model.
    :return time_taken: The time taken to train the model."""

    y_train_cate = to_categorical(y_train, num_classes)
    y_test_cate = to_categorical(y_test, num_classes)
    
    model = Sequential()
    model.add(Dense(512, input_dim=x_train.shape[1], kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(128, activity_regularizer=regularizers.l2(0.01)))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    class_weights = {}

    for i in range(num_classes):
        class_weights[i] = weights[i]

    print('\nModel summary:')
    print(model.summary())
    start = time.time()   
    
    model.fit(x_train, y_train_cate, epochs=500, batch_size=32, verbose=1, validation_data=(x_test, y_test_cate),
              shuffle=True, class_weight=class_weights)
    
    end = time.time()
    time_taken = round((end - start), 2)
    
    print('Optimization took {:.2f} seconds'.format(time_taken))
    
    predictions = model.predict(x_test)
    
    return predictions, time_taken


def dl_model_2(x_train, y_train, x_test, y_test, num_classes):
    """This function instantiates the dl model and gets the predictions from the model.

    :param x_train: The training dataset.
    :param y_train: The training dataset labels.
    :param x_test: The test dataset.
    :param y_test: The test dataset labels.
    :param num_classes: The number of classes in the dataset.
    :return: predictions, the predictions from the gp model.
    :return time_taken: The time taken to train the model."""

    y_train_cate = to_categorical(y_train, num_classes)
    y_test_cate = to_categorical(y_test, num_classes)
    
    model = Sequential()
    model.add(Dense(256, input_dim=x_train.shape[1]))
    model.add(Dense(64))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    class_weights = {}
    for i in range(num_classes):
        class_weights[i] = weights[i]

    print('\nModel summary:')
    print(model.summary())
    start = time.time()   
    
    model.fit(x_train, y_train_cate, epochs=500, batch_size=32, verbose=1, validation_data=(x_test, y_test_cate),
              shuffle=True, class_weight=class_weights)
    
    end = time.time()
    time_taken = round((end - start), 2)
    
    print('Optimization took {:.2f} seconds'.format(time_taken))
    
    predictions = model.predict(x_test)
    
    return predictions, time_taken


def evaluate(predictions, y_test, num_classes, class_map):
    """This function calculates the precision, recall, f-score and accuracy for the model's predictions.
    :param predictions: The predictions from the model.
    :param y_test: The true labels for the test dataset.
    :param num_classes: The number of classes in the test dataset.
    :param class_map: The class mapping for the dataset."""

    y_pred = np.argmax(predictions, axis=1)
    
    precision = round(precision_score(y_test, y_pred, average='weighted'), 2)
    recall = round(recall_score(y_test, y_pred, average='weighted'), 2)
    f_score = round(f1_score(y_test, y_pred, average='weighted'), 2)
    accuracy = round(accuracy_score(y_test, y_pred), 2)
    
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-Score: ", f_score)
    print("Accuracy: ", accuracy)
    
    print('\nPer Class Results:')
    print(classification_report(y_test, y_pred))
    
    confusion__matrix(y_pred, y_test, num_classes, class_map)
    
    return precision, recall, f_score, accuracy


def confusion__matrix(y_pred, y_test, num_classes, class_map):
    """This function plots the confusion matrix.

    :param y_pred: The predictions from the model.
    :param y_test: The actual labels for the test dataset.
    :param num_classes: The number of classes in the dataset.
    :param class_map: The class mapping dictionary for the dataset."""

    confusion_mtx = confusion_matrix(y_test, y_pred, normalize='true')
    classes = [i for i in range(num_classes)]
    class_names = [str(class_map[i]) for i in classes]
    plt.figure(figsize=(12, 12))
    plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
    plt.xticks(classes, class_names)
    plt.yticks(classes, class_names)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    for i in classes:
        for j in classes:
            plt.text(i, j, format(confusion_mtx[j][i], '.2f'), horizontalalignment="center",
                     color="white" if confusion_mtx[j, i] > 0.8 else "black")
    plt.show()
    

def run_models(dl_model, value):
    """This function trains the deep learning and gaussian process models increasing the number of classes each time and
    stores their performance metrics.

    :param dl_model: The deep learning model.
    :param value: The number of data objects per class.
    :return gp_accuracies: The list of accuracy values we get from the gp model.
    :return gp_time_track: The different values for the time taken by the gp model.
    :return gp_precisions: The different precision values for the gp model.
    :return gp_recalls: The different recall values for the gp model.
    :return gp_fscores: The different fscore values for the gp model
    :return dl_accuracies: The list of accuracy values we get from the dl model.
    :return dl_time_track: The different values for the time taken by the dl model.
    :return dl_precisions: The different precision values for the dl model.
    :return dl_recalls: The different recall values for the dl model.
    :return dl_fscores: The different fscore values for the dl model."""

    dataset, class_map = load_dataset(value)

    total_classes = dataset['target'].unique().shape[0]
    
    gp_accuracies = []
    gp_time_track = []
    gp_precisions = []
    gp_recalls = []
    gp_fscores = []

    dl_accuracies = []
    dl_time_track = []
    dl_precisions = []
    dl_recalls = []
    dl_fscores = []

    for i in range(3, total_classes+1):
        print('\nFor number of classes: {}\n'.format(i))
        x_train, y_train, x_test, y_test = get_data(dataset[dataset['target'].isin([c for c in range(i)])])
        num_classes = np.unique(y_train).shape[0]

        print('\nGP evaluation:')
        gp_pred, gp_time = gp_model(x_train, y_train, x_test, num_classes)
        print('\nEvaluation GP Model results:')
        gp_precision, gp_recall, gp_f_score, gp_accuracy = evaluate(gp_pred, y_test, num_classes, class_map)

        gp_accuracies.append(gp_accuracy)
        gp_time_track.append(gp_time)
        gp_precisions.append(gp_precision)
        gp_recalls.append(gp_recall)
        gp_fscores.append(gp_f_score)

        print('\nDL evaluation:')
        dl_pred, dl_time = dl_model(x_train, y_train, x_test, y_test, num_classes)
        print('\nEvaluation DL Model results:')
        dl_precision, dl_recall, dl_f_score, dl_accuracy = evaluate(dl_pred, y_test, num_classes, class_map)

        dl_accuracies.append(dl_accuracy)
        dl_time_track.append(dl_time)
        dl_precisions.append(dl_precision)
        dl_recalls.append(dl_recall)
        dl_fscores.append(dl_f_score)
    
    plot_results(total_classes, gp_accuracies, gp_time_track, gp_precisions, gp_recalls, gp_fscores,
                dl_accuracies, dl_time_track, dl_precisions, dl_recalls, dl_fscores)
    
    return gp_accuracies, gp_time_track, gp_precisions, gp_recalls, gp_fscores, dl_accuracies, dl_time_track,\
           dl_precisions, dl_recalls, dl_fscores


def plot_results(total_classes, gp_accuracies, gp_time_track, gp_precisions, gp_recalls, gp_fscores,
            dl_accuracies, dl_time_track, dl_precisions, dl_recalls, dl_fscores):
    """This function plots the different graphs comparing the performance metrics of the gp and dl models.

    :param total_classes: The total number of classes in the dataset.
    :param gp_accuracies: The list of accuracy values we get from the gp model.
    :param gp_time_track: The different values for the time taken by the gp model.
    :param gp_precisions: The different precision values for the gp model.
    :param gp_recalls: The different recall values for the gp model.
    :param gp_fscores: The different fscore values for the gp model
    :param dl_accuracies: The list of accuracy values we get from the dl model.
    :param dl_time_track: The different values for the time taken by the dl model.
    :param dl_precisions: The different precision values for the dl model.
    :param dl_recalls: The different recall values for the dl model.
    :param dl_fscores: The different fscore values for the dl model."""

    ticks = [index for index in range(3, total_classes + 1)]    
    fig, axs = plt.subplots(5, 1, figsize=(15, 30))

    axs[0].grid()
    axs[0].plot(ticks, gp_accuracies)
    axs[0].plot(ticks, dl_accuracies)  
    axs[0].set_xticks(ticks)
    axs[0].legend(['GP', 'DL'], ncol=2)
    axs[0].set_title('Accuracy')
    axs[0].set_xlabel('Number of classes')
    axs[0].set_ylabel('Accuracy Values')

    axs[1].grid()
    axs[1].plot(ticks, gp_precisions)
    axs[1].plot(ticks, dl_precisions)  
    axs[1].set_xticks(ticks)
    axs[1].legend(['GP', 'DL'], ncol=2)
    axs[1].set_title('Precision')
    axs[1].set_xlabel('Number of classes')
    axs[1].set_ylabel('Precision Values')

    axs[2].grid()
    axs[2].plot(ticks, gp_recalls)
    axs[2].plot(ticks, dl_recalls)  
    axs[2].set_xticks(ticks)
    axs[2].legend(['GP', 'DL'], ncol=2)
    axs[2].set_title('Recall')
    axs[2].set_xlabel('Number of classes')
    axs[2].set_ylabel('Recall Values')

    axs[3].grid()
    axs[3].plot(ticks, gp_fscores)
    axs[3].plot(ticks, dl_fscores)  
    axs[3].set_xticks(ticks)
    axs[3].legend(['GP', 'DL'], ncol=2)
    axs[3].set_title('F1-Score')
    axs[3].set_xlabel('Number of classes')
    axs[3].set_ylabel('F1-Score Values')

    axs[4].grid()
    axs[4].plot(ticks, gp_time_track)
    axs[4].plot(ticks, dl_time_track)  
    axs[4].set_xticks(ticks)
    axs[4].legend(['GP', 'DL'], ncol=2)
    axs[4].set_title('Time Taken')
    axs[4].set_xlabel('Number of classes')
    axs[4].set_ylabel('Time')


# Using 500 object per class Dataset
run_models(dl_model_1, 500)


# Using 1000 object per class Dataset
run_models(dl_model_2, 1000)


# Using 2000 object per class Dataset
run_models(dl_model_2, 2000)
