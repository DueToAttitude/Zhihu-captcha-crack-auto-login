# encoding: utf-8

'''分类英文验证码的CNN模型训练
代码注释参考文件cnn_train_en.py
CNN 模型组成:
    3个由卷积、激活函数、最大池化组成的层
    1个全连接层
    1个输出层
优化器:
    损失：sigmoid_cross_entropy_with_logits
    优化器：AdamOptimizer，学习率0.001

输出层:
    样本标记空间：0 (type1) 和 1 (type2)
    2个元素长度的一维向量
    [0, 1]代表type2，[1, 0]代表type1
'''

import os
from os.path import exists, isfile
from os import mkdir
from datetime import datetime
import tensorflow as tf
import numpy as np
from util_en import CAPTCHA_HEIGHT, CAPTCHA_WIDTH
from util_en import next_batch_cla, init_list

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def weight_variable(shape, w_alpha=0.01):
    initial = w_alpha * tf.random_normal(shape)
    return tf.Variable(initial, name='Weight')

def bias_variable(shape, b_alpha=0.1):
    initial = b_alpha * tf.random_normal(shape)
    return tf.Variable(initial, name='bias')

# convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# max-pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def cnn_graph(images):
    with tf.variable_scope('convolution_layer1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(images, W_conv1), b_conv1))
        h_pool1 = max_pool_2x2(h_conv1)
    with tf.variable_scope('convolution_layer2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(h_pool1, W_conv2), b_conv2))
        h_pool2 = max_pool_2x2(h_conv2)
    with tf.variable_scope('convolution_layer3'):
        W_conv3 = weight_variable([5, 5, 64, 64])
        b_conv3 = bias_variable([64])
        h_conv3 = tf.nn.relu(tf.nn.bias_add(conv2d(h_pool2, W_conv3), b_conv3))
        h_pool3 = max_pool_2x2(h_conv3)
    with tf.variable_scope('full-connected'):
        image_height = int(h_pool3.shape[1])
        image_width = int(h_pool3.shape[2])
        W_fc = weight_variable([image_width * image_height * 64, 512])
        b_fc = bias_variable([512])
        h_flat = tf.reshape(h_pool3, [-1, image_width * image_height * 64])
        h_fc = tf.nn.relu(tf.add(tf.matmul(h_flat, W_fc), b_fc))
    with tf.variable_scope('output'):
        W_out = weight_variable([512, 2])
        b_out = bias_variable([2])
        y_pred = tf.add(tf.matmul(h_fc, W_out), b_out)
        tf.summary.histogram('y_pred', y_pred)
    return y_pred
    
def optimizer_graph(y, y_pred, lr=0.001):
    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_pred))
        optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
        tf.summary.scalar('loss', loss)
    return optimizer

def accuracy_graph(y, y_pred):
    with tf.variable_scope('accuracy'):
        y_reshape = tf.argmax(y, axis=1)
        y_pred_reshape = tf.argmax(y_pred, axis=1)
        accuracy_pre = tf.equal(y_pred_reshape, y_reshape)
        accuracy = tf.reduce_mean(tf.cast(accuracy_pre, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    return accuracy

def train(train_dir, test_dir, result_dir):
    # make dir if not exists
    if not exists(result_dir):
        mkdir(result_dir)
    # input layer
    with tf.variable_scope('input'):
        x = tf.placeholder(tf.float32, [None, CAPTCHA_HEIGHT, CAPTCHA_WIDTH, 1], name='x')
        y = tf.placeholder(tf.float32, [None, 2], name='y')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    #y prediction
    y_pred = cnn_graph(x)
    #optimizer
    optimizer = optimizer_graph(y, y_pred, lr)
    #accuracy
    accuracy = accuracy_graph(y, y_pred)
    #init tensorflow
    init_list(train_dir, test_dir)
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
        batch_x, batch_y = next_batch_cla(1, 64)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, lr: 0.001})
        # print accuracy and save log every 100 times
        if step % 100 == 0:
            batch_x_test, batch_y_test = next_batch_cla(2, 100)
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
            # store training result if acc larger than acc_rate
            if acc > acc_rate:
                model_path = os.getcwd() + os.sep + result_dir + os.sep + str(acc_rate) + '_' + str(acc) + "captcha.model"
                saver.save(sess, model_path, global_step=step)
                if acc_rate < 0.99: acc_rate += 0.01
                # if acc_rate > 0.99: break
        step += 1
    sess.close()

if __name__ == '__main__':
    train('train4', 'train3', 'result/en3')
