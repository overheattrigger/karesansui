#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import cv2
import numpy as np
import math
import tensorflow as tf
import random
import time
tf.logging.set_verbosity(tf.logging.INFO)
NUM_CLASSES = 18
IMAGE_ROWS = 150
IMAGE_COLS = 200
IMAGE_PIXELS = IMAGE_ROWS*IMAGE_COLS*3

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train', 'train.txt', 'File name of train data')
flags.DEFINE_string('test', 'test.txt', 'File name of train data')
flags.DEFINE_string('train_dir', '/tmp/data', 'Directory to put the training data.')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 30, 'Batch size'
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 128, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden2', 32, 'Number of steps to run trainer.')

def inference(images, hidden1_units, hidden2_units):
  """Build the MNIST model up to where it may be used for inference.

  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                            stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    logits = tf.matmul(hidden2, weights) + biases
  return logits

def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss

def training(loss, learning_rate):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))

def do_eval(sess, eval_correct, test_data, test_label):
  """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.

  true_count += sess.run(eval_correct, feed_dict={x: test_data, y_:test_label})
  precision = float(true_count) / len(test_label)
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (len(test_label), true_count, precision))

if __name__ == '__main__':

    # ファイルを開く
    print ("open training data")
    f = open('train.txt', 'r')
    #データを入れる配列
    train_list = []
    i = 0
    for line in f:
        # 改行を除いてスペース区切りにする
        line = line.rstrip()
        l = line.split()
        train_list.append((l[0], l[1]))
    f.close()

    print("open test data")
    f = open('test.txt', 'r')
    test_image = []
    test_label = []
    test_list = []
    for line in f:
        line = line.rstrip()
        l = line.split()
        img = cv2.imread(l[0])
        # img = cv2.resize(img, (28, 28))
        test_image.append(img.flatten().astype(np.float32)/255.0)
        tmp = np.zeros(NUM_CLASSES)
        # tmp[int(l[1])] = 1
        test_list.append(l)
        test_label.append(l[1])
    test_image = np.asarray(test_image)
    test_label = np.asarray(test_label)
    f.close()
    
    # Create the model
    x = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, IMAGE_PIXELS))
    y_ = tf.placeholder(tf.int32, shape=(FLAGS.batch_size))
    logits = inference(x, FLAGS.hidden1, FLAGS.hidden2)
    loss_ = loss(logits, y_)
    train_op = training(loss_, FLAGS.learning_rate)
    eval_correct = evaluation(logits, y_)

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    # train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # 保存の準備
    saver = tf.train.Saver()
    # sessionの作成
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # TensorBoardで表示する値の設定
    # summary_op = tf.merge_all_summaries()
    # summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph_def)

    # Train
    for step in range(FLAGS.max_steps):
        start_time = time.time()
        print ("start train! %d" % step )
        l = random.sample(train_list, FLAGS.batch_size)
        train_image = []
        train_label = []
        
        for i in l:
            img = cv2.imread(i[0])

            # 一列にした後、0-1のfloat値にする
            train_image.append(img.flatten().astype(np.float32)/255.0)
            # ラベルを1-of-k方式で用意する
            tmp = np.zeros(NUM_CLASSES)
            # tmp[int(i[1])] = 1
            train_label.append(i[1])
            #numpy形式に変換
        train_image = np.asarray(train_image)
        train_label = np.asarray(train_label)
        _, loss_value = sess.run([train_op, loss_], feed_dict={x: train_image, y_: train_label})
        duration = time.time() - start_time
        if step % 100 == 0:
            # Print status to stdout.
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Save a checkpoint and evaluate the model periodically.
        if (step + 1) % 100 == 0 or (step + 1) == FLAGS.max_steps:
            l = random.sample(test_list, FLAGS.batch_size)
            test_image = []
            test_label = []
            for i in l:
                img = cv2.imread(i[0])

                # 一列にした後、0-1のfloat値にする
                test_image.append(img.flatten().astype(np.float32)/255.0)
                # ラベルを1-of-k方式で用意する
                tmp = np.zeros(NUM_CLASSES)
                # tmp[int(i[1])] = 1
                test_label.append(i[1])
                #numpy形式に変換
            test_image = np.asarray(test_image)
            test_label = np.asarray(test_label)
            # Evaluate against the test set.
            print('Test Data Eval:')
            do_eval(sess, eval_correct, test_image, test_label)
            # Update the events file.
            # summary_str = sess.run(summary, feed_dict=feed_dict)
            # summary_writer.add_summary(summary_str, step)
            # summary_writer.flush()

    # Test trained model
    # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print(sess.run(accuracy, feed_dict={x: test_image,
    #                                     y_: test_label}))

    # 最終的なモデルを保存
    save_path = saver.save(sess, "model.ckpt", global_step=FLAGS.max_steps)
