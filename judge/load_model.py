#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import cv2
import numpy as np
import math
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('trainDir', '../../TrainData', 'Directory for train data')
flags.DEFINE_string('testDir', '../../TestData', 'Direcotry for test data')
flags.DEFINE_string('labelFile', './label.csv', 'output filename for label listt')
flags.DEFINE_string('resultFile', './result.csv', 'output filename for result')
flags.DEFINE_integer('maxSteps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batchSize', 50, 'Batch size, Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('imageRows', 150, 'row size of input data')
flags.DEFINE_integer('imageCols', 200, 'col size of input data')
flags.DEFINE_integer('imageChannels', 3, 'channel of input data')
IMAGE_PIXELS = FLAGS.imageRows*FLAGS.imageCols*FLAGS.imageChannels

def make_label_list():
  f = open(FLAGS.labelFile, 'r')
  idList = []
  labelList = []
  for line in f:
    line.rstrip
    l = line.split(',')
    idList.append(l[0])
    labelList.append(l[1])
  return idList, labelList

def read_images(targetDir):
  images = []
  dirPath = targetDir + '/'
  files = os.listdir(dirPath)
  fileNameList = []
  for f in files:
    filePath = dirPath + '/' + f
    img = cv2.imread(filePath)
    # 一列にした後、0-1のfloat値にする
    images.append(img.flatten().astype(np.float32)/255.0)
    # ラベルを1-of-k方式で用意する
    fileNameList.append(filePath)
  # numpy形式に変換
  images = np.asarray(images)
  return images, fileNameList

if __name__ == '__main__':
  labelList, _ = make_label_list()
  
  targetImage, fileNameList = read_images(FLAGS.testDir)

  labelNum = len(labelList)
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

  saver.restore(sess, "./model.ckpt")
  print("Model restored.")
  result = sess.run(y, feed_dict={x: targetImage})

  f = open(FLAGS.resultFile, 'w')
  for i in range(0, len(result)):
    index = np.argmax(result[i])
    f.write('%s, %s, %d\n' % (fileNameList[i], labelList[index], index))
