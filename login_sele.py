# encoding: utf-8

'''知乎模拟登陆
使用Python库selenium进行模拟登陆
登陆操作会打开浏览器，知乎正常登录过程一般不需要验证码。
为了演示识别验证码的过程，在不需要验证码时，会自动在账号密码输入框中输入错误的信息，
点击登录后，会提示出错，此时一般就会有验证码出现。
程序一检测到验证码图片，就会自动开始识别，识别完成后，会自动清空输入框内容，并输入正确的账号密码验证码等信息。
此时程序挂起等待，点击登录即可完成整个过程
'''
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
import tensorflow as tf
from time import sleep
from bs4 import BeautifulSoup
import re
import base64
from PIL import Image
import cnn_test_en
import cnn_test_cn
import cnn_test_en_cla

username = 'username'
password = 'password'

def decode_img(html):
    '''抓取HTML页面中的验证码图片并解码
    :param html: HTML页面
    :return img_type: 中文或英文
    :return img: array类型的验证码图片
    '''
    # 使用正则表达式抓取网页中的验证码图片和验证码类型信息
    img_element = re.findall(r'<img alt="图形验证码" class="Captcha-(.+?)" data-tooltip="看不清楚？换一张" src="data:image/jpg;base64,(.+?)"/>', 
        html, re.S)[0]
    # 列表第一个数据表示验证码类型，英文为"englishImg"，中文为"chineseImg"
    img_type = img_element[0]
    # 列表第二个数据表示验证码图片，图片为base64编码
    img_base64 = img_element[1].replace(r'\n', '')
    # 将验证码图片保存下来
    with open('./captcha.jpg', 'wb') as f:
        # base64解码
        f.write(base64.b64decode(img_base64))
    # 打开验证码图片为Image类型
    img = Image.open('./captcha.jpg')
    # img.show()
    return img_type, img


def login_sele():
    '''知乎模拟登陆
    '''
    input('<Begin>')
    # 不打开浏览器运行
    # chrome_options = Options()
    # chrome_options.add_argument("--headless")
    # 初始化driver
    driver = webdriver.Chrome()
    # driver = webdriver.Edge()
    # 初始化ActionChains
    actions = ActionChains(driver)
    # 打开知乎登录界面
    driver.get('https://www.zhihu.com/signup')
    # 定位到注册切换按钮
    signup_switch_bt = driver.find_element_by_xpath('//*[@id="root"]/div/main/div/div/div/div[2]/div[2]/span')
    # 知乎默认打开注册界面，若注册切换按钮上的文字为登录，说明当前处于注册界面，此时点击此按钮以进入登录界面
    if signup_switch_bt.text == '登录':
        signup_switch_bt.click()
    # 定位账号输入框
    uname_textfield = driver.find_element_by_xpath('//*[@id="root"]/div/main/div/div/div/div[2]/div[1]/form/div[1]/div[2]/div[1]/input')
    # 定位密码输入框
    pwd_textfield = driver.find_element_by_xpath('//*[@id="root"]/div/main/div/div/div/div[2]/div[1]/form/div[2]/div/div[1]/input')
    # 定位登录按钮
    signup_bt = driver.find_element_by_xpath('//*[@id="root"]/div/main/div/div/div/div[2]/div[1]/form/button')
    # 输入错误的账户密码信息
    uname_textfield.send_keys('wrong_username')
    pwd_textfield.send_keys('wrongpassword')
    # 自动检测页面是否有验证码图片，每隔1秒检测一次，60次后超时
    i = 0
    timeout = 60
    while True:
        i += 1
        try:
            # 获取HTML页面
            html = BeautifulSoup(driver.page_source, "html5lib").prettify()
            # 使用正则表达式抓取图片
            html_find = re.findall(r'<img alt="图形验证码" class="Captcha-(.+?)" data-tooltip="看不清楚？换一张" src="data:image/jpg;base64,(.+?)"/>'
                , html, re.S)
            # 若超过60次或验证码不为空，则跳出循环
            if i > timeout or html_find[0][1] != 'null':
                break
            # 等待1秒
            sleep(1)
        except Exception as e:
            # 捕获到异常，原因是当前不在登录界面，因此html_find为空，调用html_find[0][1]后抛出数组越界的异常
            # 此时将继续检测页面，等待1秒后继续循环
            if i > timeout:
                break
            sleep(1)
            continue
    # 等待1秒以使得页面加载完成
    sleep(1.5)
    # 清空账户输入框并输入正确的账户
    uname_textfield.clear()
    uname_textfield.send_keys(username)
    # 清空密码输入框并输入正确的密码
    pwd_textfield.clear()
    pwd_textfield.send_keys(password)
    # 开始识别验证码
    if i <= timeout:
        print('Begin Recognization!')
        try:
            # g = tf.Graph()
            # 定位验证码图片
            img_element = driver.find_element_by_xpath(u"//img[@alt='图形验证码']")
            # 解码验证码，获得验证码类型和验证码图片
            img_type, img = decode_img(BeautifulSoup(driver.page_source, "html5lib").prettify())
            # 类型为中文验证码
            if img_type == 'chineseImg':
                # with g.as_default():
                # 初始化中文验证码识别模型
                cnn_cn = cnn_test_cn.CnnTest('result/cn')
                # 识别中文验证码，结果为表示倒立汉字位置的坐标
                points = cnn_cn.cnn_test_single(img)
                print('Position:', points)
                # 自动在验证码上点击相应的坐标
                for p in points:
                    # 将鼠标移动到验证码图片左上角加上偏置的位置上，该偏置为倒立汉字的坐标
                    actions.move_to_element_with_offset(img_element, p[0], p[1])
                    # 点击鼠标
                    actions.click()
                actions.perform()
            # 类型为英文验证码
            elif img_type == 'englishImg':
                # with g.as_default():
                # 初始化英文类型识别模型
                cnn_cla = cnn_test_en_cla.CnnTest('result/en_cla')
                # 识别图片样式，0表示样式1,1表示样式2
                type = cnn_cla.cnn_test_single(img)
                # g1 = tf.Graph()
                # 根据识别结果初始化不同不同类型英文验证码识别模型
                if type == 0:
                    # with g1.as_default():
                    cnn_en = cnn_test_en.CnnTest('result/en1')
                else:
                    # with g1.as_default():
                    cnn_en = cnn_test_en.CnnTest('result/en2')
                # 识别英文验证码，结果为验证码字符串
                capt = cnn_en.cnn_test_single(img)
                print('Text:', capt)
                # 定位验证码输入框并输入验证码
                driver.find_element_by_name("captcha").send_keys(capt)
            else: print('No image found!')
        except Exception as e:
            print(e)
    input('<End>')
    driver.close()


if __name__ == '__main__':
    while True:
        login_sele()
