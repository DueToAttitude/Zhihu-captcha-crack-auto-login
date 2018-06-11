# encoding: utf-8

'''英文验证码的CNN模型训练
CNN 模型组成:
    3个由卷积、批标准化、激活函数、最大池化组成的层
    1个全连接层
    1个输出层
优化器:
    损失：sigmoid_cross_entropy_with_logits
    优化器：AdamOptimizer，学习率0.001

输出层:
    样本标记空间：0-9, a-z (36)
    验证码字符串长度：4
    输出36 * 4 = 144 长度的一维向量
    每36个元素代表一个字符
    把字符在样本标记空间的位置对应的元素设为1，其余设为0
    例如 [1, 0, 0...] 表示字符 '0'
'''

import os
from os.path import exists, isfile
from os import mkdir
from datetime import datetime
import tensorflow as tf
import numpy as np
from util_en import CAPTCHA_LIST, CAPTCHA_LEN, CAPTCHA_HEIGHT, CAPTCHA_WIDTH
from util_en import next_batch, init_list

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def weight_variable(shape, w_alpha=0.01):
    '''使用正态分布初始化weight
    :param shape: weight矩阵的形状
    :param w_alpha: 对应的权值
    :return: 初始化完成的weight
    '''
    initial = w_alpha * tf.random_normal(shape)
    return tf.Variable(initial, name='Weight')

def bias_variable(shape, b_alpha=0.1):
    '''使用正太分布初始化bias
    '''
    initial = b_alpha * tf.random_normal(shape)
    return tf.Variable(initial, name='bias')

def conv2d(x, W):
    '''卷积，步长为1，为1，padding模式为"SAME"
    :param x: 卷积的输入
    :param W: 卷积核矩阵
    :return: 卷积的输出
    '''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    '''最大池化，步长为2，为1，padding模式为"SAME"
    :param x: 最大池化的输入
    :return: 输出
    '''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# batch-normalization
def batch_normalization(x):
    '''批标准化
    :param x:
    :return:
    '''
    with tf.variable_scope('batch_normalization'):
        x_mean, x_var = tf.nn.moments(x, [0, 1, 2])
        scale = tf.Variable(tf.ones([1]), name='scale')
        offset = tf.Variable(tf.zeros([1]), name='offset')
        return tf.nn.batch_normalization(x, x_mean, x_var, offset, scale, 0.001)

def cnn_graph(images):
    '''CNN模型
    :param images: 多张图片构成的array类型的矩阵
    :return: 模型对输入的图片验证码的预测
    '''
    x = batch_normalization(images)
    # 卷积网络的一个基本层
    with tf.variable_scope('convolution_layer1'):
        # 初始化weight和bias
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        # 卷积
        h_conv1 = tf.nn.bias_add(conv2d(x, W_conv1), b_conv1)
        # 批标准化
        h_bn1 = batch_normalization(h_conv1)
        # 激活函数
        h_conv1 = tf.nn.relu(h_bn1)
        # 最大池化
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
    # 全连接层
    with tf.variable_scope('full-connected'):
        # 获取经过上述代码的处理之后图片的高度和宽度
        image_height = int(h_pool3.shape[1])
        image_width = int(h_pool3.shape[2])
        # 初始化weight和bias
        W_fc = weight_variable([image_width * image_height * 64, 1024])
        b_fc = bias_variable([1024])
        # 将数据“展平”
        h_flat = tf.reshape(h_pool3, [-1, image_width * image_height * 64])
        h_fc = tf.nn.relu(tf.add(tf.matmul(h_flat, W_fc), b_fc))
    # 输出层
    with tf.variable_scope('output'):
        W_out = weight_variable([1024, CAPTCHA_LEN * len(CAPTCHA_LIST)])
        b_out = bias_variable([CAPTCHA_LEN * len(CAPTCHA_LIST)])
        y_pred = tf.add(tf.matmul(h_fc, W_out), b_out)
        tf.summary.histogram('y_pred', y_pred)
    return y_pred
    
def optimizer_graph(y, y_pred, lr=0.001):
    '''优化器
    :param y: 真实值
    :param y_pred: 预测值
    :param lr: 学习率
    :return: 优化器
    '''
    with tf.variable_scope('loss'):
        # loss的计算
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_pred))
        # 使用Adam优化器最小化loss，学习率设为lr
        optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
        tf.summary.scalar('loss', loss)
    return optimizer

def accuracy_graph(y, y_pred):
    '''计算准确率
    :param y: 真实值
    :param y_pred: 预测值
    :return: 准确率
    '''
    with tf.variable_scope('accuracy'):
        # 计算表示真实值的一维向量每36个元素中最大的元素所在的位置，该位置即为字符在字符集中的位置
        y_reshape = tf.argmax(tf.reshape(y, [-1, CAPTCHA_LEN, len(CAPTCHA_LIST)]), axis=2)
        # 计算表示预测值的一维向量每36个元素中最大的元素所在的位置
        y_pred_reshape = tf.argmax(tf.reshape(y_pred, [-1, CAPTCHA_LEN, len(CAPTCHA_LIST)]), axis=2)
        accuracy_pre = tf.equal(y_pred_reshape, y_reshape)
        # 计算均值，结果为单个字符预测的准确率
        accuracy = tf.reduce_mean(tf.cast(accuracy_pre, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    return accuracy

def train(train_dir, test_dir, result_dir):
    '''CNN训练
    :param train_dir: 训练集文件夹
    :param test_dir: 测试集文件夹
    :param result_dir: 存放结果的文件夹
    :return:
    '''
    # 如果存放结果的文件夹不存在，则创建一个
    if not exists(result_dir):
        mkdir(result_dir)
    # 定义输入
    with tf.variable_scope('input'):
        x = tf.placeholder(tf.float32, [None, CAPTCHA_HEIGHT, CAPTCHA_WIDTH, 1], name='x')
        y = tf.placeholder(tf.float32, [None, CAPTCHA_LEN * len(CAPTCHA_LIST)], name='y')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    # 通过CNN模型预测输出
    y_pred = cnn_graph(x)
    # 使用真实值和预测值得到优化器
    optimizer = optimizer_graph(y, y_pred, lr)
    # 计算单个字符的准确率
    accuracy = accuracy_graph(y, y_pred)
    # 初始化util_en中的全局变量
    init_list(train_dir, test_dir)
    # 初始化TensorFlow
    saver = tf.train.Saver()
    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # 从文件中恢复已经存储的模型，以从这个模型开始继续训练
    # saver.restore(sess, tf.train.latest_checkpoint(result_dir))
    writer = tf.summary.FileWriter(result_dir + '/log', sess.graph)
    merge = tf.summary.merge_all()
    # 若准确率大于这个预定值，则存储模型到本地文件
    acc_rate = 0.95
    # 训练的步数
    step = 0
    # 若准确率大于0.8，则存储模型到本地，整个过程中对于所有大于0.8的训练结果只存储第一个结果
    store = True
    # 开始训练
    while True:
        # 获取训练集中的数据，图片矩阵和真实值，每个batch大小为64
        batch_x, batch_y = next_batch(1, 64)
        # 一步训练
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, lr: 0.001})
        # 每训练100部，打印或者存储训练结果
        if step % 100 == 0:
            # 获取测试集中的数据，图片矩阵和真实值，每个batch大小为100
            batch_x_test, batch_y_test = next_batch(2, 100)
            # 获取准确率
            acc, result = sess.run([accuracy, merge], feed_dict={x: batch_x_test, y: batch_y_test})
            stdout = str(datetime.now().strftime('%c')) + '    step: ' + str(step) + '  accuracy: ' + str(acc)
            # 输出准确率信息
            print(stdout)
            # 保存log文件
            with open(result_dir + '/output.log', 'a') as file:
                file.write(stdout + '\n')
            writer.add_summary(result, step)
            # 若准确率大于0.8，则存储模型到本地
            if store and acc >= 0.8:
                # 保存模型
                model_path = os.getcwd() + os.sep + result_dir + os.sep + str(0.80) + '_' + str(acc) + "captcha.model"
                saver.save(sess, model_path, global_step=step)
                # 设为false，后续训练中准确率大于0.8的结果将不会被存储
                store = False
            # 如果准确率大于acc_rate，则保存模型
            if acc > acc_rate:
                # 保存模型
                model_path = os.getcwd() + os.sep + result_dir + os.sep + str(acc_rate) + '_' + str(acc) + "captcha.model"
                saver.save(sess, model_path, global_step=step)
                # 若acc_rate小于0.99，则加上0.01，这样准确率大于0.95，0.96，0.97，0.98和0.99时各分别只存储一次
                if acc_rate < 0.99: acc_rate += 0.01
                # if acc_rate > 0.99: break
        step += 1
    sess.close()

if __name__ == '__main__':
    train('train2', 'test2', 'result/en2_1')
