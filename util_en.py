# encoding: utf-8

'''英文验证码基础函数和变量
CAPTCHA_LIST：定义了验证码字符集
CAPTCHA_LEN：定义验证码字符长度
CAPTCHA_WIDTH：定义验证码图片宽度
CAPTCHA_LENGTH：定义验证码图片长度
text2vec：验证码字符串转化为验证码向量形式

    验证码向量长度为 len(CAPTCHA_LIST) * CAPTHCA_LEN，每 len(CAPTCHA_LIST) 个向量元素表示一个字符，
    字符在字符集的位置对应的元素记为1，其他元素记为0。例如字符0可表示为
    1 0 0 0...0 0(len(CAPTCHA_LIST) - 1个0)
'''

import numpy as np
from random import choice
import os
from PIL import Image
import time

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
LOW_CASE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
UP_CASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
           'V', 'W', 'X', 'Y', 'Z']
CAPTCHA_LIST = NUMBER + LOW_CASE + ['_']
CAPTCHA_LEN = 4
CAPTCHA_HEIGHT = 60
CAPTCHA_WIDTH = 150

# 全局变量
# 训练集所有文件名字所构成的列表
train_list = []
# 测试集所有文件名字所构成的列表
test_list = []
# 训练集文件夹
train_dir = ''
# 测试集文件夹
test_dir = ''

def text2vec(text, captcha_len=CAPTCHA_LEN, captcha_list=CAPTCHA_LIST):
    '''将验证码字符串转化为向量形式
    :param text: 验证码字符串
    :return: 验证码向量
    '''
    # 获取验证码字符串长度
    text_len = len(text)
    # 长度超出cpatcha_len，返回异常
    if text_len > captcha_len:
        raise ValueError('验证码字符串长度不能超过' + str(captcha_len))
    # 初始化向量，元素个数为 captcha_len * len(captcha_list)
    vector = np.zeros(captcha_len * len(captcha_list))
    # 以字符为单位逐个处理向量
    for i in range(text_len):
        # 将与字符串在字符集的位置对应的元素置1
        vector[captcha_list.index(text[i]) + i * len(captcha_list)] = 1
    return vector


def vec2text(vec, captcha_list=CAPTCHA_LIST, size=CAPTCHA_LEN):
    '''将验证码向量转化为字符串
    :param vec: 验证码向量
    :return: 验证码字符串
    '''
    # 一维向量转化为 size * len(captcha_list) 的二维矩阵
    vec = np.reshape(vec, [size, len(captcha_list)])
    # 求出二维矩阵最后一个维度最大元素的位置
    vec_idx = np.argmax(vec, axis=1)
    # 最大元素的位置即为字符串在字符集中的位置，根据最大元素的位置求出字符串
    text_list = [captcha_list[i] for i in vec_idx]
    # 将列表转化为字符串
    return ''.join(text_list)


def gen_list(dir):
    '''求出dir文件夹中所有文件的名字所构成的列表
    :param dir:
    :return: 所有文件的列表
    '''
    captcha_text_list = []
    for root, dirnames, images_name in os.walk(dir):
        for image_name in images_name:
            captcha_text_list.append(image_name)
    return captcha_text_list


def gen_captcha_text_and_image(dir, captcha_text_list):
    '''在dir文件夹中所有图片里随机选取一个图片,用于构建英文验证码识别的批次，要求图片名字的前四位为验证码字符串
    :param dir:
    :param captcha_text_list: 所有图片构成的列表
    :return catpcha text: 验证码图片字符串
    :return captcha image: 验证码图片
    '''
    # 从所有图片构成的列表中随机选取一个元素
    image_text = choice(captcha_text_list)
    # 打开该图片并将其转化为矩阵的形式
    image = np.array(Image.open(dir + "/" + image_text))
    return image_text[:4].lower(), image


def gen_captcha_text_and_image_cla(dir, captcha_text_list):
    '''和上一个函数作用相同，用于构建英文验证码分类的批次，要求图片名字的第一位表示图片样式
    '''
    cla_vec = []
    image_text = choice(captcha_text_list)
    image = np.array(Image.open(dir + "/" + image_text))
    # 若第一位为字符"1"，则输出为向量[1, 0]
    if image_text[0] == '1':
        cla_vec = [1, 0]
    else:
        cla_vec = [0, 1]
    return cla_vec, image


def next_batch(type=1, batch_size=64, width=CAPTCHA_WIDTH, height=CAPTCHA_HEIGHT):
    '''构建一个批次的数据集，用于英文验证码识别模型的训练
    :param type: 1表示训练集，2表示测试集
    :param batch_siee: 一个批次的样本容量
    :param width:
    :param height:
    :return: 一个批次的数据
    '''
    # 初始化为0矩阵
    batch_x = np.zeros([batch_size, height, width, 1])
    batch_y = np.zeros([batch_size, CAPTCHA_LEN * len(CAPTCHA_LIST)])
    for i in range(batch_size):
        if type == 1:
            # type为1，从训练集获取一张图片的数据
            text, image = gen_captcha_text_and_image(train_dir, train_list)
        else:
            # type为其他，从测试集获取一张图片的数据
            text, image = gen_captcha_text_and_image(test_dir, test_list)
        # image = convert2gray(image)
        batch_x[i, :] = image.reshape(height, width, 1)
        batch_y[i, :] = text2vec(text)
    return batch_x, batch_y


def next_batch_cla(type=1, batch_size=64, width=CAPTCHA_WIDTH, height=CAPTCHA_HEIGHT):
    '''和next_batch函数相似，只是这个函数用于分类模型的训练
    '''
    batch_x = np.zeros([batch_size, height, width, 1])
    batch_y = np.zeros([batch_size, 2])
    for i in range(batch_size):
        if type == 1:
            # get training data
            cla_vec, image = gen_captcha_text_and_image_cla(train_dir, train_list)
        else:
            # get testing data
            cla_vec, image = gen_captcha_text_and_image_cla(test_dir, test_list)
        # image = convert2gray(image)
        batch_x[i, :] = image.reshape(height, width, 1)
        batch_y[i, :] = cla_vec
    return batch_x, batch_y


def init_list(atrain_dir, atest_dir):
    '''初始化训练集和测试集
    初始化训练集文件列表和测试集文件列表，读取训练集和测试集中的所有文件，将它们的名字分别保存在相应的列表中
    这样，只需读取一次文件列表，每次调用gen_captcha_text_and_image_cla函数和gen_captcha_text_and_image函数就不用再去读取文件列表
    可以大大加快程序的运行，特别是每个文件夹里包含大量文件的时候
    '''
    global train_list, test_list, train_dir, test_dir
    train_list = gen_list(atrain_dir)
    test_list = gen_list(atest_dir)
    train_dir = atrain_dir
    test_dir = atest_dir




