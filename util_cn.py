# encoding: utf-8

'''中文验证码基础函数和变量
CAPTCHA_LIST：定义了验证码字符集
CAPTCHA_LEN：定义验证码字符长度
CAPTCHA_WIDTH：定义验证码图片宽度
CAPTCHA_LENGTH：定义验证码图片长度

    验证码向量长度为 2*7=14，每2个向量元素表示一个字符，
    [0,1]表示倒立字符，[1,0]表示正常字符
'''

import numpy as np
import random
from captcha_generate_cn import ImageChar


CAPTCHA_LEN = 7
CAPTCHA_HEIGHT = 44
CAPTCHA_WIDTH = 200


def list2vec(list):
    '''将列表转化为一维向量，其中列表中的每个元素表示倒立汉字的位置，从0开始标记第一个位置
    比如[1,3]表示7个汉字中第2个和第4个汉字为倒立汉字
    '''
    # 0值初始化一维向量
    vec = np.zeros(CAPTCHA_LEN * 2)
    # 将一维向量设为[1 0 1 0 1 0...1 0]，表示所有汉字都是正常的状态
    for i in range(CAPTCHA_LEN):
        vec[i * 2] = 1
    # 将倒立汉字对应的位置由[1 0]修改成[0 1]
    for num in list:
        if num < CAPTCHA_LEN:
            vec[num * 2] = 0
            vec[num * 2 + 1] = 1
    return vec


def vec2list(vec):
    '''将一维向量转化为表示倒立汉字位置的列表，从0开始标记第一个位置，比如[1,3]表示7个汉字中第2个和第4个汉字为倒立汉字
    '''
    # 初始化列表为空
    list = []
    # 遍历
    for i in range(len(vec) // 2):
        # 若表示同一个汉字的2个元素置为[0 1]，也即第一个元素小于第二个元素，则表示该位置为倒立的汉字
        if vec[2 * i] < vec[2 * i + 1]:
            # 将这个位置添加进列表
            list += [i]
    return list


def gen_captcha_list_and_image():
    '''随机生成一张中文验证码图片
    '''
    while True:
        try:
            ic = ImageChar()
            # 倒立汉字个数随机取1或2
            num_flip = random.randint(1,2)
            image, num_flip_list, char_list = ic.randChinese(CAPTCHA_LEN, num_flip)
            return image, num_flip_list
        # 若抛出异常，原因是随机生成的区位码在GB2312编码中为空
        except:
            # 抛出异常，则再进行一次生成验证码的代码
            continue


def next_batch(batch_size=64, width=CAPTCHA_WIDTH, height=CAPTCHA_HEIGHT):
    '''构建一个批次的数据集，用于中文验证码识别模型的训练
    :param batch_size: 一个批次的样本容量
    :param width:
    :param height:
    :return: 一个批次的数据
    '''
    batch_x = np.zeros([batch_size, height, width, 1])
    batch_y = np.zeros([batch_size, CAPTCHA_LEN * 2])
    for i in range(batch_size):
        image, list = gen_captcha_list_and_image()
        batch_x[i, :] = image.reshape(height, width, 1)
        batch_y[i, :] = list2vec(list)
    return batch_x, batch_y


if __name__ == '__main__':
    print(list2vec([2,4]))
    print(vec2list([0,1,0,1,1,0,1,0,0,1]))


