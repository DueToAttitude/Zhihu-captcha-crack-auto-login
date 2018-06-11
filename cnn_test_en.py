# encoding: utf-8

'''英文验证码的CNN模型测试
通过输入的图片预测出验证码
'''

import tensorflow as tf
import os
from os.path import join
from shutil import move, copyfile
import numpy as np
from PIL import Image
import time
from cnn_train_en import cnn_graph, accuracy_graph
from util_en import gen_captcha_text_and_image
from util_en import vec2text, text2vec, gen_list, next_batch, init_list
from util_en import CAPTCHA_LIST, CAPTCHA_WIDTH, CAPTCHA_HEIGHT, CAPTCHA_LEN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def cnn_test_all(test_dir, result_dir):
    '''读取测试文件夹的所有验证码图片，并使用模型进行预测
    :param test_dir: 测试文件夹
    :param result_dir: 存储模型的文件夹
    :return: 所有验证码的真实值以及预测值
    '''
    # 一个批次的大小
    batch_size = batch_size1 = 100
    # 获取测试文件夹中所有的验证码文件名字
    captcha_text_list = gen_list(test_dir)
    # 计算训练批次的数量
    batch_num = (len(captcha_text_list) + batch_size - 1) // batch_size
    # 初始化
    batch_x = np.zeros([len(captcha_text_list), CAPTCHA_HEIGHT, CAPTCHA_WIDTH, 1])
    batch_y = np.zeros([len(captcha_text_list), CAPTCHA_LEN * len(CAPTCHA_LIST)])
    # 从测试文件夹中读取所有验证码，将验证码图片构成的矩阵和验证码真实值分别存储在batch_x和batch_y中
    for i in np.arange(len(captcha_text_list)):
        # 打开一个验证码图片
        image = np.array(Image.open(join(test_dir, captcha_text_list[i])))
        image = image.reshape(CAPTCHA_HEIGHT, CAPTCHA_WIDTH, 1)
        batch_x[i, :] = image
        # 验证码图片的名字前四位由该验证码字符串构成
        batch_y[i, :] = text2vec(captcha_text_list[i][:4].lower())
    # 初始化TensorFlow的输入
    x = tf.placeholder(tf.float32, [None, CAPTCHA_HEIGHT, CAPTCHA_WIDTH, 1])
    y = tf.placeholder(tf.float32, [None, CAPTCHA_LEN * len(CAPTCHA_LIST)])
    # 通过卷积神经网络得到预测值
    y_pred = cnn_graph(x)
    # 计算准确率
    accuracy = accuracy_graph(y, y_pred)
    saver = tf.train.Saver()
    # 存储预测值的变量
    predictions = np.array([])
    # 记录准确率的变量
    acc = 0
    with tf.Session() as sess:
        # 从模型中恢复，模型位于result_dir文件夹中
        saver.restore(sess, tf.train.latest_checkpoint(result_dir))
        # 每次使用batch_size个样本进行测试，目的是防止内存耗尽
        for i in np.arange(batch_num):
            # 进行到最后一个批次的时候，剩下的样本容量大小可能小于batch_size，使用batch_size1记录这个大小
            if len(captcha_text_list) < batch_size * (i + 1):
                batch_size1 = len(captcha_text_list) - batch_size * i
            # 第一个批次需要单独处理，主要目的是给predictions变量赋一个值，后面的批次就可以直接将结果连接到这次的结果之上
            if i == 0:
                predictions = sess.run(y_pred, feed_dict={x: batch_x[i * batch_size:i * batch_size + batch_size1]})
            else:
                predictions = np.append(predictions, sess.run(y_pred, feed_dict={x: batch_x[i * batch_size:i * batch_size + batch_size1]}), 0)
            # 当前批次计算出来的准确率乘以batch_size1的值累加进acc变量中，最终得到的acc表示预测正确的个数
            acc += sess.run(accuracy, feed_dict={x: batch_x[i * batch_size:i * batch_size + batch_size1],
                y: batch_y[i * batch_size:i * batch_size + batch_size1]}) * batch_size1
    # 除以总数得到最终的准确率
    acc = acc / len(captcha_text_list)
    print('accuracy(single character):', acc)
    return batch_y, predictions


class CnnTest(object):
    '''模型测试对象
    其他函数通过新建此对象调用模型识别功能
    先进行模型的初始化，在调用测试函数，此测试适用于单个样本的输入
    '''
    def __init__(self, result_dir):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # 初始化TensorFlow的输入
            self.x = tf.placeholder(tf.float32, [None, CAPTCHA_HEIGHT, CAPTCHA_WIDTH, 1])
            # 预测模型的构建，通过调用CNN框架得到预测值
            self.y_pred = cnn_graph(self.x)
            self.saver = tf.train.Saver()
            self.sess = tf.Session()
            # 从指定文件夹中恢复模型
            self.saver.restore(self.sess, tf.train.latest_checkpoint(result_dir))

    def cnn_test_single(self, image):
        '''识别单个验证码图片
        :param image: 待识别的验证码图片
        :return: 验证码字符串
        '''
        # 将验证码图片转化为矩阵
        image = np.array(image)
        image = image.reshape(1, CAPTCHA_HEIGHT, CAPTCHA_WIDTH, 1)
        # 进行预测
        y_pred_temp = self.sess.run(self.y_pred, feed_dict={self.x: image})
        # 模型预测值为一维向量，将其转化为字符串
        return [vec2text(i) for i in y_pred_temp][0]
