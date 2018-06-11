# encoding: utf-8

'''英文验证码分类的CNN模型测试
通过输入的图片预测出验证码类别
'''

import tensorflow as tf
import os
from shutil import move, copyfile
import numpy as np
from PIL import Image
import time
import random
from cnn_train_en_cla import cnn_graph, accuracy_graph
from util_en import gen_captcha_text_and_image_cla
from util_en import gen_list, next_batch_cla, init_list
from util_en import CAPTCHA_WIDTH, CAPTCHA_HEIGHT

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def cnn_test_all(test_dir, result_dir):
    '''
    :param image: numpy array(-1, captcha_height, captcha_width, 1) 
    :return: captcha text (list of str)
    '''
    batch_size = batch_size1 = 100
    # get all testing set
    captcha_text_list = gen_list(test_dir)
    # captcha_text_list = random.sample(captcha_text_list, 8000)
    batch_num = (len(captcha_text_list) + batch_size - 1) // batch_size
    # batch_num = 1
    batch_x = np.zeros([len(captcha_text_list), CAPTCHA_HEIGHT, CAPTCHA_WIDTH, 1])
    batch_y = np.zeros([len(captcha_text_list), 2])
    for i in np.arange(len(captcha_text_list)):
        image = np.array(Image.open(test_dir + '/' + captcha_text_list[i]))
        image = image.reshape(CAPTCHA_HEIGHT, CAPTCHA_WIDTH, 1)
        batch_x[i, :] = image
        if captcha_text_list[i][0] == '1':
            batch_y[i, :] = [1, 0]
        else:
            batch_y[i, :] = [0, 1]
    # print(len(batch_x))
    # CNN test
    x = tf.placeholder(tf.float32, [None, CAPTCHA_HEIGHT, CAPTCHA_WIDTH, 1])
    y = tf.placeholder(tf.float32, [None, 2])
    y_pred = cnn_graph(x)
    accuracy = accuracy_graph(y, y_pred)
    saver = tf.train.Saver()
    predictions = np.array([])
    acc = 0
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(result_dir))
        # predictions, acc = sess.run([y_pred, accuracy], feed_dict={x: batch_x[:batch_size], y: batch_y[:batch_size]})
        for i in np.arange(batch_num):
            if len(captcha_text_list) < batch_size * (i + 1):
                batch_size1 = len(captcha_text_list) - batch_size * i
            if i == 0:
                predictions = sess.run(y_pred, feed_dict={x: batch_x[i * batch_size:i * batch_size + batch_size1]})
            else:
                predictions = np.append(predictions, sess.run(y_pred, feed_dict={x: batch_x[i * batch_size:i * batch_size + batch_size1]}), 0)
            accd = sess.run(accuracy, feed_dict={x: batch_x[i * batch_size:i * batch_size + batch_size1],
                y: batch_y[i * batch_size:i * batch_size + batch_size1]})
            print(accd)
            acc += accd * batch_size1
    acc = acc / len(captcha_text_list)
    return np.argmax(batch_y, 1), np.argmax(predictions, 1)


class CnnTest(object):
    def __init__(self, result_dir):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, [None, CAPTCHA_HEIGHT, CAPTCHA_WIDTH, 1])
            # self.y = tf.placeholder(tf.float32, [None, 2])
            self.y_pred = cnn_graph(self.x)
            self.saver = tf.train.Saver()
            self.sess = tf.Session()
            self.saver.restore(self.sess, tf.train.latest_checkpoint(result_dir))

    def cnn_test_single(self, image):
        image = np.array(image)
        image = image.reshape(1, CAPTCHA_HEIGHT, CAPTCHA_WIDTH, 1)
        
        y_pred_temp = self.sess.run(self.y_pred, feed_dict={self.x: image})
        return np.argmax(y_pred_temp, axis=1)[0]

