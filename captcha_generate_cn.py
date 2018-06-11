# encoding: utf-8

'''使用Pillow库模仿知乎中文验证码自己画图
知乎验证码图片类型为GIF，识别时将其先转化为模式"L"，因此我们模仿的参照就是转化为模式"L"的图片

知乎验证码的特征主要由：
    1 验证码大小为宽400个像素，高88个像素；
    2 中文字符字体为华文楷体粗字体；
    3 左边距在0-13个像素之间随机取值，字体间隔在-6-4个像素之间随机取值；
    4 字符在y轴方向上大致在中间位置，在-5-5个像素的上下波动范围内随机取值；
    5 每个像素只有一个维度的数据，表现为灰度的图片，空白处的灰度为249，字符处的灰度为85；
    6 每个字符随机旋转-15-15度。

'''
from PIL import Image,ImageDraw,ImageFont
import random
import numpy as np

CAPTCHA_HEIGHT = 88
CAPTCHA_WIDTH = 400

class RandomChar():
    @staticmethod
    def GB2312():
        '''使用gb23121编码从3755个一级汉字中随机选取一个字符
        对于gb2312编码，并不是所有的区位都有一个字符与之对应，若刚好随机选取到这些位置，则会抛出异常，需要捕获并处理这个异常
        :return: 一个汉字字符
        '''
        # 高8位编码
        head = random.randint(0xB0, 0xD7)
        # 低8位编码的前4位
        body = random.randint(0xA, 0xF)
        # 低8位编码的后4位
        tail = random.randint(0, 0xF)
        val = ( head << 8 ) | (body << 4) | tail
        # 将int类型数据转化为bytes数据
        bs = val.to_bytes((val.bit_length() + 7) // 8, 'big')
        # 编码
        str = bs.decode('GB2312')
        # 返回汉字字符
        return str

class ImageChar():
    def __init__(self, size = (CAPTCHA_WIDTH, CAPTCHA_HEIGHT),  # 图片大小
    bgColor = (249,249,249,255),                                # 背景颜色，设为249
    # 请自行在Windows安装这个字体，或者选择更加适合的字体
    fontPath = 'C:\\Windows\\Fonts\\HUAWENKAITI-BOLD-1.TTF',    # 字体类型：华文楷体粗字体
    fontSize = random.randint(60, 61),                          # 字体大小，随机取[60, 61]中的一个
    fontColor = (85,85,85,255),                                 # 字体颜色，设为85
    charSize = (64, 68),                                        # 字体画布大小
    charColor = (0,0,0,0),                                      # 字体画布背景颜色，设为透明，这样在黏贴到原图片时不会覆盖
    zoom = 20):                                                 # 放大倍数，先将图片放大一定倍数，将字符画上去之后再缩小相应倍数，这样的目的是使得字符边缘变得平滑锐利，减少毛边
        self.size = tuple(np.array(size) * zoom)
        self.bgColor = bgColor
        self.fontPath = fontPath
        self.fontSize = fontSize * zoom
        self.fontColor = fontColor
        self.charSize = tuple(np.array(charSize) * zoom)
        self.charColor = charColor
        self.zoom = zoom
        # 初始化字体，包括字体路径和大小
        self.font = ImageFont.truetype(self.fontPath, self.fontSize)
        # 最初构建的图片模式为"RGBA"，画好字符之后再转化为模式"L"
        # 这样做的目的是，"RGBA"模式有alpha通道的概念，可以将字体画布背景颜色设为透明，以便合成时颜色不会叠加干扰
        self.image = Image.new('RGBA', self.size, self.bgColor) 

    def drawTextV2(self, pos, txt, angle):
        '''新建字符画布，并画出字符，最后和原图片结合
        :param pos: 字符画布结合在原图片上的位置
        :param txt: 需要画上的字符
        :param angle: 旋转角度
        '''
        image=Image.new('RGBA', self.charSize, self.charColor)
        draw = ImageDraw.Draw(image)
        draw.text((1 * self.zoom, -11 * self.zoom), txt, font=self.font, fill=self.fontColor)
        w=image.rotate(angle,  expand=0)
        self.image.alpha_composite(w, dest=pos)
        del draw

    def randChinese(self, num, num_flip):
        '''生成中文验证码图片
        :param num: 一张图片中中文字符个数，默认为7个
        :param num_flip: 倒立中文字符个数
        :return image: 矩阵形式的验证码图片
        :return num_flip_list: 倒立字符位置列表
        :return char_list: 字符列表
        '''
        # left padding = start + gap，为两个均匀分布的随机变量的相加
        # left padding 范围[0, 13], 当取值在范围[3, 10]时概率最大，其他取值越远离该范围概率越小
        # gap指相邻字符的间距
        gap = random.randint(-6, 4)
        start = random.randint(6, 9)
        # 表示倒立字符位置的列表
        num_flip_list = random.sample(range(num), num_flip)
        # 表示倒立字符的列表
        char_list = []
        # 增加num个中文字符到char_list中，步骤为：
        # 1 随机选取一个中文字符
        # 2 新建字符画布，将字符画在画布上
        # 3 将字符画布和原图片合并，x和y表示字符画布在原图片的位置
        for i in range(num):
            # 随机选取一个中文字符
            char = RandomChar().GB2312()
            # 添加进字符列表
            char_list.append(char)
            # 设置字符画布在原图片位置的x轴坐标
            x = max(0, (start + ((self.fontSize // self.zoom - 60) * 2 + 49) * i + gap + gap * i) * self.zoom)
            # 设置字符画布在原图片位置的y轴坐标
            # upper padding 范围在[4, 25], 当取值在范围[10, 19]时概率最大，其他取值越远离该范围概率越小
            # 不直接使用均匀分布的原因是会使得字符上下波动程度过大
            y = random.randint(11, 17)
            y += random.randint(-7, 8)
            y *= self.zoom
            # 字符随机旋转的角度
            # 旋转角度范围在[-15, 15], 当取值在范围[-5, 5]时概率最大，其他取值越远离该范围概率越小
            angle = random.randint(-5, 5)
            angle += random.randint(-10, 10)
            # 对于需要倒立的汉字字符，则旋转角度增加180度
            if i in num_flip_list:
                angle += 180
            # 通过上述得到的参数，画出字符
            self.drawTextV2((x, y), char, angle)
        # 由于验证码图片最终显示在浏览器的大小为(200, 44),缩小的一半。
        # 所以这里讲图片缩小zoom*2倍，最终为(200, 44)，最终作为CNN的输入图片也是这个大小
        # 缩放时候，将图片转化为模式"L"
        self.image = self.image.resize(tuple(np.array(self.size) // (self.zoom))).convert('L')
        # 返回矩阵形式的验证码图片，倒立字符位置列表和字符列表
        return np.array(self.image), num_flip_list, char_list

    def save(self, path):
        '''保存验证码图片到本地
        :param path: 本地位置
        '''
        self.image.save(path)


if __name__ == '__main__':
    # 表示抛出字符不存在异常的次数
    err_num = 0
    for i in range(1):
        try:
            ic = ImageChar()
            # 倒立字符个数随机从(1, 2)中选取
            num_flip = random.randint(1,2)
            # 生成中文验证码图片
            _, num_flip_list, char_list = ic.randChinese(7, num_flip)
            # 保存到本地，以字符列表和倒立字符列表命名图片
            ic.save(''.join(char_list)+'_'+''.join(str(i) for i in num_flip_list)+".png")
        except:
            # 选取字符抛出异常，则异常次数加1
            err_num += 1
            continue
    # 输出异常次数
    print('error:', err_num)




