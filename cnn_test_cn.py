# encoding: utf-8

'''中文验证码的CNN模型测试
通过输入的图片预测出验证码
'''

import tensorflow as tf
import os
import json
import random
import numpy as np
from PIL import Image
import time
from cnn_train_cn import cnn_graph, accuracy_graph
from util_cn import gen_captcha_list_and_image
from util_cn import vec2list, list2vec, next_batch
from util_cn import CAPTCHA_WIDTH, CAPTCHA_HEIGHT, CAPTCHA_LEN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
        # 计算字符在验证码图片中的边界
        padding = get_padding(np.array(image.resize((CAPTCHA_WIDTH, CAPTCHA_HEIGHT))))
        # 处理验证码图片，使其格式与CNN的输入格式一致
        image = image.convert('L').resize((CAPTCHA_WIDTH, CAPTCHA_HEIGHT))
        image = np.array(image)
        image = image.reshape(1, CAPTCHA_HEIGHT, CAPTCHA_WIDTH, 1)
        # 使用CNN得到预测值
        y_pred_temp = self.sess.run(self.y_pred, feed_dict={self.x: image})
        # 倒立汉字位置列表
        num_flip_list = [vec2list(i) for i in y_pred_temp][0]
        print('num_flip_list:', [i + 1 for i in num_flip_list])
        points = []
        # 根据倒立汉字位置列表和字符边界计算倒立汉字具体的坐标
        for num in num_flip_list:
            points += [[padding[0] + (num + 0.5) * (padding[1] - padding[0]) / 7,
            padding[2] + (padding[3] - padding[2]) / 2 + random.uniform(-0.5, 0.5)]]
        # capt = json.dumps({'img_size': [200, 44], 'input_points': [[i[0], i[1]] for i in points]})
        # return capt, num_flip_list
        return points


def get_padding(image):
    '''计算验证码图片中字符上下左右边界所在位置
    :param image: 输入的验证码图片，
    此图片的类型为GIF，转化为矩阵形式后，图片空白背景处值为0，字符笔画处值为1
    '''
    # 初始化上下左右边界
    left = upper = 0
    right = CAPTCHA_WIDTH
    lower = CAPTCHA_HEIGHT

    # 计算左边界，从图片的左上角像素开始，依次从上到下，从左到右遍历图片
    for i in range(CAPTCHA_WIDTH):
        for j in range(CAPTCHA_HEIGHT):
            # 当遇到第一个值为1的像素时，即遇到字符笔画时，标记左边界为此像素的x轴坐标，再跳出循环
            if image[j, i] != 0:
                left = i
                break
        # 上述跳出循环只能跳出内层的循环，这段代码用于跳出外层的循环
        # 如果内层循环正常退出，则执行else子句，外层的break被跳过
        # 如果内层循环为break退出，则跳过else子句，执行下一个break
        else:
            continue
        break
    # 计算右边界，从图片的右上角像素开始，依次从上到下，从右到左遍历图片
    for i in range(CAPTCHA_WIDTH):
        for j in range(CAPTCHA_HEIGHT):
            if image[j, CAPTCHA_WIDTH - i - 1] != 0:
                right = CAPTCHA_WIDTH - i - 1
                break
        else:
            continue
        break
    # 计算上边界，从图片的左上角像素开始，依次从左到右，从上到下遍历图片
    for j in range(CAPTCHA_HEIGHT):
        for i in range(CAPTCHA_WIDTH):
            if image[j, i] != 0:
                upper = j
                break
        else:
            continue
        break
    # 计算下边界，从图片的左下角像素开始，依次从左到右，从下到上遍历图片
    for j in range(CAPTCHA_HEIGHT):
        for i in range(CAPTCHA_WIDTH):
            if image[CAPTCHA_HEIGHT - j - 1, i] != 0:
                lower = CAPTCHA_HEIGHT - j - 1
                break
        else:
            continue
        break
    return left, right, upper, lower

if __name__ == '__main__':
    pass
