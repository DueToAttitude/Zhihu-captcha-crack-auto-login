# encoding: utf-8

'''获取知乎英文和中文验证码
英文验证码：
    1 通过爬取验证码api下载验证码图片
    2 图片提交若快平台，自动识别
    3 识别结果使用和步骤1相同的session提交知乎验证码api，验证识别结果正确与否
中文验证码：
    1 直接爬取下载图片至本地 
'''

import requests
import time
import re
import base64
import json
from rk import RClient
from os.path import exists, join
from os import mkdir

# 请求验证码的headers
HEADERS = {
    'Connection': 'keep-alive',
    'Host': 'www.zhihu.com',
    'Referer': 'https://www.zhihu.com/signup',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'
    }
# 知乎验证码api
LOGIN_URL = 'https://www.zhihu.com/signup'
# 知乎英文验证码api
CAPTCHA_API_EN = 'https://www.zhihu.com/api/v3/oauth/captcha?lang=en'
# 知乎中文验证码api
CAPTCHA_API_CN = 'https://www.zhihu.com/api/v3/oauth/captcha?lang=cn'
# 若快账户信息
username = 'username'   # 账户名
password = 'password'   # 密码
soft_id = 'soft_id'     # soft_id
soft_key = 'soft_key'   # soft_key

def get_captcha_pic_en(dir_succ, dir_fail):
    '''获取知乎英文验证码
    将识别成功和失败的图片分别保存至不同文件夹
    :param dir_succ: 识别成功文件所保存的文件夹
    :param dir_fail:
    :return text: 返回验证码识别结果
    :return is_succ: 是否识别成功标志
    '''
    # 步骤1 各种初始化操作
    # 若文件夹不存在，则创建文件夹
    if not exists(dir_succ):
        mkdir(dir_succ)
    if not exists(dir_fail):
        mkdir(dir_fail)
    # 获取session，下载验证码图片和提交识别结果必须使用同一个session
    session = requests.session()
    # 设置该session的headers
    session.headers = HEADERS.copy()
    # 更新headers中的两个参数，其中"authorization"为固定值，"X-Xsrftaken"需要先请求一次知乎登录api，在response的headers中提取出来
    session.headers.update({
        'authorization': 'oauth c3cef7c66a1843f8b3a9e6a1e3160e20',
        'X-Xsrftoken': get_token(session)
    })

    # 步骤2 下载验证码图片
    # 下载验证码之前需要使用GET方法请求一次验证码api，来判断此次登录请求是否需要验证码，不管返回结果是True或者False，都可以通过PUT方法下载验证码图片
    # 此步骤是必须的，不可删除，若删除，则下载验证码失败
    session.get(CAPTCHA_API_EN)
    # 下载验证码图片
    resp = session.put(CAPTCHA_API_EN)
    # 验证码图片以base64编码存储在response中，先使用正则表达式提取图片部分，再使用base64解码
    img_base64 = re.findall(r'"img_base64":"(.+)"', resp.text, re.S)
    # 若正则提取结果不为空，则先将换行符替换掉
    if len(img_base64) > 0:
        img_base64 = img_base64[0].replace(r'\n', '')
    # 使用base64将图片解码成二进制形式
    img = base64.b64decode(img_base64)

    # 步骤3 使用若快打码平台识别验证码
    # 使用若快打码平台识别验证码，此步骤先进行初始化，需要提供若快账户和密码，以及soft_id和soft_key，详情见若快官网www.ruokuai.com
    rc = RClient(username, password, soft_id, soft_key)
    # 将二进制图片文件作为参数输入
    resp = rc.rk_create(img)
    # 识别结果，类型为字符串
    text = str(resp['Result']).lower()
    
    # 步骤4 识别结果提交知乎，判断识别是否正确
    # 识别结果成功与否标志
    is_succ = True
    # 使用POST方法提交识别结果
    resp = session.post(CAPTCHA_API_EN, data={'input_text': text})
    # 若返回的状态码为201，表明识别成功
    if resp.status_code == 201:
        is_succ = True
        # 将识别成功的验证码保存至dir_succ文件夹，文件使用识别结果的字符串命名
        with open(join(dir_succ, text + '.gif'), 'wb') as f:
            f.write(img)
    else:
        is_succ = False
        # 将识别成功的验证码保存至dir_fail文件夹，文件使用识别结果的字符串命名
        with open(join(dir_fail, text + '.gif'), 'wb') as f:
            f.write(img)
    # 返回识别结果字符串以及成功与否标志
    return text, is_succ


def get_captcha_pic_cn(fromdir):
    '''获取知乎中文验证码
    具体逻辑和英文验证码的相同
    :param fromdir:
    '''
    session = requests.session()
    session.headers = HEADERS.copy()
    session.headers.update({
        'authorization': 'oauth c3cef7c66a1843f8b3a9e6a1e3160e20',
        'X-Xsrftoken': get_token(session)
    })

    resp = session.get(CAPTCHA_API_CN)
    resp = session.put(CAPTCHA_API_CN)
    img_base64 = re.findall(r'"img_base64":"(.+)"', resp.text, re.S)
    if len(img_base64) > 0:
        img_base64 = img_base64[0].replace(r'\n', '')
    img = base64.b64decode(img_base64)
    with open(fromdir, 'wb') as f:
        f.write(img)


def get_token(session=None):
    """请求登录api，从response中提取"X-Xsrftakon"信息作为后续请求的headers
    :param session:
    :return: _xsrf
    """
    resp = session.get(LOGIN_URL)
    # 需要的信息位于response的headers，headers的Set-Cookie，Set-Cookie的_xsrf中
    token = re.findall(r'_xsrf=([\w|-]+)', resp.headers.get('Set-Cookie'))[0]
    return token


if __name__ == '__main__':
    # 表示循环次数
    i = 1
    while True:
        text, is_succ = get_captcha_pic_en('train_succ', 'train_fail')
        print(str(i) + ': ' + str(is_succ) + ', ' + text)
        i += 1
        # 表示最大的图片获取数量
        if i > 5:
            break