"""
RNN model for crop classification
"""

import os, sys, logging, gc, shutil
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow import nn
from tensorflow.python import debug as tf_debug
from os.path import join
import settings

tf.logging.set_verbosity(tf.logging.INFO)


"""
Before:
'Corn': 0, 'Soybeans': 1, 'Cotton': 2, 'SpringWheat': 3, 'DurumWheat': 3, 'WinterWheat': 6, 'Other': 6

After:
'Corn': 0, 'Soybeans': 1, 'Cotton': 2, 'Wheat': 3, 'Other': 4
"""
"""
RNN model for crop classification
"""


def RNN(x, weights, biases):

    print(num_hidden)
    x = tf.unstack(x, timesteps, 1)

    lstm_cell_qx = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    lstm_cell_qx = rnn.DropoutWrapper(lstm_cell_qx, output_keep_prob=(1 - dropout))

    lstm_cell_hx = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    lstm_cell_hx = rnn.DropoutWrapper(lstm_cell_hx, output_keep_prob=(1 - dropout))

    outputs, _, _, = rnn.static_bidirectional_rnn(
        lstm_cell_qx, lstm_cell_hx, x, dtype=tf.float32
    )

    return tf.matmul(outputs[-1], weights["out"]) + biases["out"]


def prepare_data(features, labels):
    labels = labels[:, np.newaxis]
    labels[labels == 6] = 4
    data = np.concatenate([features, labels], axis=1)
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    data = data[idx, :]

    # Corn
    data_Corn = data[data[:, -1] == 0, :]
    print(data_Corn.shape)
    # Soybeans
    data_Soybeans = data[data[:, -1] == 1, :]
    print(data_Soybeans.shape)
    # Cotton to 50%
    data_Cotton = data[data[:, -1] == 2, :]
    length = int(len(data_Cotton) / 2.0)
    data_Cotton = data_Cotton[0:length, :]
    print(data_Cotton.shape)
    # Wheat to 33%
    data_Wheat = data[data[:, -1] == 3, :]
    length = int(len(data_Wheat) / 3.0)
    data_Wheat = data_Wheat[0:length, :]
    print(data_Wheat.shape)
    # Other to 80%
    data_Other = data[data[:, -1] == 4, :]
    length = int(len(data_Other) / 1.25)
    data_Other = data_Other[0:length, :]
    print(data_Other.shape)

    data = np.concatenate(
        [data_Corn, data_Soybeans, data_Cotton, data_Wheat, data_Other], axis=0
    )
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    data = data[idx, :]

    features = data[:, 0:-1]
    labels = data[:, -1]
    return features, labels


if __name__ == "__main__":
    # NOW here we define traning and model parameters
    # Training Parameters
    learning_rate = 0.01
    training_steps = 300
    batch_size = 256

    # Network Parameters
    num_input = 6
    timesteps = 123
    num_classes = 5
    dropout = 0.5
    num_hidden = 512

    display_step = 10

    model_dir = "./model_dir"

    for key in settings.file_pair.keys():
        tf.reset_default_graph()
        print("Model:" + key)
        print("File:" + settings.file_pair[key][1])
        print("Initializing data")
        features_train = np.load(settings.file_pair[key][1])["features"]
        labels_train = np.load(settings.file_pair[key][1])["labels"]
        features_train, labels_train = prepare_data(features_train, labels_train)

        features_test = np.load(settings.file_pair[key][0])["features"]
        labels_test = np.load(settings.file_pair[key][0])["labels"]
        features_test, labels_test = prepare_data(features_test, labels_test)

        # convert nan to 0, get rid of nan
        features_train = np.nan_to_num(features_train)

        # reshape training testing data into matrix
        features_train = features_train.reshape((-1, num_input, timesteps))
        features_train = features_train.transpose(0, 2, 1)

        # random shuffle data
        rand_array = np.arange(features_train.shape[0])
        np.random.shuffle(rand_array)
        features_train = features_train[rand_array]
        labels_train = labels_train[rand_array]

        labels_train = tf.one_hot(labels_train, num_classes)

        features_test = features_test.reshape((-1, num_input, timesteps))
        features_test = features_test.transpose(0, 2, 1)
        labels_test = tf.one_hot(labels_test, num_classes)

        print("TRAINING TOTAL %d" % features_train.shape[0])
        print("TESTING TOTAL %d" % features_test.shape[0])

        # now create computational graph
        num_batch = int(features_train.shape[0] / batch_size)

        # tf Graph input
        X = tf.placeholder(
            "float", [None, timesteps, num_input], name="input_placeholder"
        )
        Y = tf.placeholder("float", [None, num_classes])

        # Define weights
        weights = {"out": tf.Variable(tf.random_normal([num_hidden * 2, num_classes]))}
        biases = {"out": tf.Variable(tf.random_normal([num_classes]))}

        logits = RNN(X, weights, biases)
        prediction = tf.nn.softmax(logits)
        predict_tag = tf.identity(prediction, "predict_the_fuck")

        # Define loss and optimizer
        loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y)
        )
        optimizer = tf.train.AdagradOptimizer(
            learning_rate=learning_rate, initial_accumulator_value=0.1
        )
        train_op = optimizer.minimize(loss_op)

        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        conf = tf.confusion_matrix(
            tf.argmax(labels_test, 1), tf.argmax(prediction, 1), num_classes
        )

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            labels_train = sess.run(labels_train)

            for step in range(1, training_steps + 1):
                print("Step", str(step))

                for b in range(num_batch + 1):
                    # print(b)
                    batch_x, batch_y = (
                        features_train[b * batch_size : (b + 1) * batch_size],
                        labels_train[b * batch_size : (b + 1) * batch_size],
                    )

                    # Run optimization op (backprop)
                    sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                if step % display_step == 0 or step == 1:
                    batch_loss = []
                    batch_acc = []
                    # Calculate batch loss and accuracy
                    for b in range(num_batch):
                        batch_x, batch_y = (
                            features_train[b * batch_size : (b + 1) * batch_size],
                            labels_train[b * batch_size : (b + 1) * batch_size],
                        )
                        loss, acc = sess.run(
                            [loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y}
                        )
                        batch_loss.append(loss)
                        batch_acc.append(acc)
                    batch_loss_av = sum(batch_loss) / float(len(batch_loss))
                    batch_acc_av = sum(batch_acc) / float(len(batch_acc))
                    print(
                        "Step "
                        + str(step)
                        + ", Minibatch Loss= "
                        + "{:.4f}".format(batch_loss_av)
                        + ", Training Accuracy= "
                        + "{:.3f}".format(batch_acc_av)
                    )

                    # Calculate accuracy for testing data
                    test_data = features_test
                    test_label = sess.run(labels_test)
                    test_accuracy = sess.run(
                        accuracy, feed_dict={X: test_data, Y: test_label}
                    )
                    print("Testing Accuracy:", test_accuracy)
                    confusion = sess.run(conf, feed_dict={X: test_data, Y: test_label})
                    print(confusion)
                    f = open("res.txt", "a")
                    f.write(key + "\n")
                    f.write(str(test_accuracy) + "\n")
                    f.write(str(confusion) + "\n\n")
                    f.close()
                    saver = tf.train.Saver()

                    shutil.rmtree(model_dir)
                    os.makedirs(model_dir)

                    saver.save(sess, join(model_dir, key))
