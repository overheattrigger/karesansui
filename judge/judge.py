#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import sys
import cv2
import numpy as np
import math
import tensorflow as tf
import random
import time
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('trainDir', '../../TrainData', 'Directory for train data')
flags.DEFINE_string('testDir', '../../TestData', 'Direcotry for test data')
flags.DEFINE_string('labelFile', './label.csv', 'output filename for label listt')
flags.DEFINE_integer('maxSteps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batchSize', 50, 'Batch size, Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('imageRows', 150, 'row size of input data')
flags.DEFINE_integer('imageCols', 200, 'col size of input data')
flags.DEFINE_integer('imageChannels', 3, 'channel of input data')
IMAGE_PIXELS = FLAGS.imageRows*FLAGS.imageCols*FLAGS.imageChannels

def make_train_list():
  files = os.listdir(FLAGS.trainDir)
  list = []
  for file in files:
    index = 0
    for i in range(0, len(list)):
      if int(file) < int(list[i]):
        index = i
        break
      if i == len(list) - 1:
        index = len(list)
        break
    list.insert(index, file)
  f = open(FLAGS.labelFile, 'w')
  for i in range(0, len(list)):
    f.write('%s,%d\n' % (list[i], i))

  return list

def read_images(targetDir, labelList):
  images = []
  labels = []
  labelNum = len(labelList)
  for i in range(0, len(labelList)):
    dir = labelList[i]
    dirPath = targetDir + '/' + dir
    files = os.listdir(dirPath)
    for f in files:
      filePath = dirPath + '/' + f
      img = cv2.imread(filePath)
      # 一列にした後、0-1のfloat値にする
      images.append(img.flatten().astype(np.float32)/255.0)
      # ラベルを1-of-k方式で用意する
      tmp = np.zeros(labelNum)
      tmp[i] = 1
      labels.append(tmp)
  # numpy形式に変換
  images = np.asarray(images)
  labels = np.asarray(labels)
  print ('image number = %d' % len(images))
  return images, labels

def next_batch(images, labels):
  index = random.sample(range(0, len(images)), FLAGS.batchSize)
  imageBatch = []
  labelBatch = []
  for i in index:
    imageBatch.append(images[i])
    labelBatch.append(labels[i])
  return imageBatch, labelBatch
  
if __name__ == '__main__':

  trainList = make_train_list()
  
  trainImage, trainLabel = read_images(FLAGS.trainDir, trainList)
  testImage, testLabel = read_images(FLAGS.testDir, trainList)

  labelNum = len(trainList)
  # Create the model
  x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])
  W = tf.Variable(tf.zeros([IMAGE_PIXELS, labelNum]))
  b = tf.Variable(tf.zeros([labelNum]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, labelNum])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  # 保存の準備
  saver = tf.train.Saver()
  # sessionの作成
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  start = time.time()
  for step in range(FLAGS.maxSteps):
    batch_xs, batch_ys = next_batch(trainImage, trainLabel)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if ((step % 100) == 0):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      print ("step %d, training accuracy %g"%(step, sess.run(accuracy, feed_dict={x: testImage, y_: testLabel})))

  elapsed_time = time.time() - start
  print (("elapsed_time:{0}".format(elapsed_time)) + "[sec]")
  # 最終的なモデルを保存
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print ("step %d, training accuracy %g"%(step, sess.run(accuracy, feed_dict={x: testImage, y_: testLabel})))
  save_path = saver.save(sess, "model.ckpt")
