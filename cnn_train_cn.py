# encoding: utf-8

'''中文验证码CNN模型训练
代码注释参考文件cnn_train_en.py
该模型识别中文字符中的倒立字符的位置
CNN 模型组成:
    3个由卷积、批标准化、激活函数、最大池化组成的层
    1个全连接层
    1个输出层
优化器:
    损失：sigmoid_cross_entropy_with_logits
    优化器：AdamOptimizer，学习率0.001

输出层:
    样本标记空间：0 (正常) 和 1 (倒立)
    汉字字符串长度为7，每个汉字是否倒立的状态由2个元素表示，构成14个元素的一维向量
    [0, 1]代表倒立，[1, 0]代表正常
'''

import os
from os.path import exists, isfile
from datetime import datetime
import tensorflow as tf
import numpy as np
from util_cn import CAPTCHA_LEN, CAPTCHA_HEIGHT, CAPTCHA_WIDTH
from util_cn import next_batch


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def weight_variable(shape, w_alpha=0.01):
    # initial = tf.truncated_normal(shape, stddev=0.1)
    initial = w_alpha * tf.random_normal(shape)
    return tf.Variable(initial, name='Weight')

def bias_variable(shape, b_alpha=0.1):
    # initial = tf.constant(0.1, shape=shape)
    initial = b_alpha * tf.random_normal(shape)
    return tf.Variable(initial, name='bias')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def batch_normalization(x):
    with tf.variable_scope('batch_normalization'):
        x_mean, x_var = tf.nn.moments(x, [0, 1, 2])
        scale = tf.Variable(tf.ones([1]), name='scale')
        offset = tf.Variable(tf.zeros([1]), name='offset')
        return tf.nn.batch_normalization(x, x_mean, x_var, offset, scale, 0.001)

def cnn_graph(images):
    x = batch_normalization(images)
    with tf.variable_scope('convolution_layer1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.bias_add(conv2d(x, W_conv1), b_conv1)
        h_bn1 = batch_normalization(h_conv1)
        h_conv1 = tf.nn.relu(h_bn1)
        h_pool1 = max_pool_2x2(h_conv1)
    with tf.variable_scope('convolution_layer2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.bias_add(conv2d(h_pool1, W_conv2), b_conv2)
        h_bn2 = batch_normalization(h_conv2)
        h_conv2 = tf.nn.relu(h_bn2)
        h_pool2 = max_pool_2x2(h_conv2)
    with tf.variable_scope('convolution_layer3'):
        W_conv3 = weight_variable([5, 5, 64, 64])
        b_conv3 = bias_variable([64])
        h_conv3 = tf.nn.bias_add(conv2d(h_pool2, W_conv3), b_conv3)
        h_bn3 = batch_normalization(h_conv3)
        h_conv3 = tf.nn.relu(h_bn3)
        h_pool3 = max_pool_2x2(h_conv3)
    with tf.variable_scope('full-connected'):
        image_height = int(h_pool3.shape[1])
        image_width = int(h_pool3.shape[2])
        W_fc = weight_variable([image_width * image_height * 64, 512])
        b_fc = bias_variable([512])
        h_flat = tf.reshape(h_pool3, [-1, image_width * image_height * 64])
        h_fc = tf.nn.relu(tf.add(tf.matmul(h_flat, W_fc), b_fc))
    with tf.variable_scope('output'):
        W_out = weight_variable([512, CAPTCHA_LEN * 2])
        b_out = bias_variable([CAPTCHA_LEN * 2])
        y_pred = tf.add(tf.matmul(h_fc, W_out), b_out)
        # tf.summary.histogram('y_pred', y_pred)
    return y_pred
    
def optimizer_graph(y, y_pred, lr=0.001):
    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_pred))
        optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
        tf.summary.scalar('loss', loss)
    return optimizer

def accuracy_graph(y, y_pred):
    with tf.variable_scope('accuracy'):
        y_reshape = tf.argmax(tf.reshape(y, [-1, CAPTCHA_LEN, 2]), axis=2)
        y_pred_reshape = tf.argmax(tf.reshape(y_pred, [-1, CAPTCHA_LEN, 2]), axis=2)
        accuracy_pre = tf.equal(y_pred_reshape, y_reshape)
        accuracy = tf.reduce_mean(tf.cast(tf.floor(tf.reduce_mean(tf.cast(accuracy_pre, tf.float32), axis=1)), tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    return accuracy

def train(result_dir):
    # make dir if not exists
    if not exists(result_dir) or isfile(result_dir):
        os.mkdir(result_dir)
    # input layer
    with tf.variable_scope('input'):
        x = tf.placeholder(tf.float32, [None, CAPTCHA_HEIGHT, CAPTCHA_WIDTH, 1], name='x')
        y = tf.placeholder(tf.float32, [None, CAPTCHA_LEN * 2], name='y')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    #prediction
    y_pred = cnn_graph(x)
    #optimizer
    optimizer = optimizer_graph(y, y_pred, lr)
    #accuracy
    accuracy = accuracy_graph(y, y_pred)
    #init tensorflow
    saver = tf.train.Saver()
    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # saver.restore(sess, tf.train.latest_checkpoint(result_dir))
    writer = tf.summary.FileWriter(result_dir + '/log', sess.graph)
    merge = tf.summary.merge_all()
    #training begin
    acc_rate = 0.95
    step = 0
    # store training result once when accuracy larger than 0.8
    store = True
    while 1:
        batch_x, batch_y = next_batch(64)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, lr: 0.001})
        # print accuracy and save log every 100 times
        if step % 100 == 0:
            batch_x_test, batch_y_test = next_batch(100)
            acc, result = sess.run([accuracy, merge], feed_dict={x: batch_x_test, y: batch_y_test})
            stdout = str(datetime.now().strftime('%c')) + '    step: ' + str(step) + '  accuracy: ' + str(acc)
            print(stdout)
            with open(result_dir + '/output.log', 'a') as file:
                file.write(stdout + '\n')
            writer.add_summary(result, step)
            # store training result once when accuracy larger than 0.8
            if store and acc >= 0.8:
                model_path = os.getcwd() + os.sep + result_dir + os.sep + str(0.80) + '_' + str(acc) + "captcha.model"
                saver.save(sess, model_path, global_step=step)
                store = False
            if acc > acc_rate:
                model_path = os.getcwd() + os.sep + result_dir + os.sep + str(acc_rate) + '_' + str(acc) + "captcha.model"
                saver.save(sess, model_path, global_step=step)
                if acc_rate < 0.99: acc_rate += 0.01
                # if acc_rate > 0.99: break
        step += 1
    sess.close()

if __name__ == '__main__':
    train('result/cn2')
