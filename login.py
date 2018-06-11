# encoding: utf-8

'''Sumilate login to Zhihu
The process:
    1 get value _sxrf and x-uuid
     GET: https://www.zhihu.com/signup
    2 check whether it need to get captcha, if True, continue, if False, set captcha to None and skip to step 5
     GET: https://www.zhihu.com/api/v3/oauth/captcha?lang=cn
    3 download captcha picture
     PUT: https://www.zhihu.com/api/v3/oauth/captcha?lang=cn
    4 post captcha text to confirm, if True, continue, if False, goto step 2
     POST: https://www.zhihu.com/api/v3/oauth/captcha?lang=cn
    5 post (user, password, catpcha, signature) to login
     POST: https://www.zhihu.com/signup
'''

import requests
import time
import re
import base64
import hmac
import hashlib
import json
import matplotlib.pyplot as plt
from http import cookiejar
from PIL import Image
from bs4 import BeautifulSoup
import cnn_test_en
import cnn_test_cn
import cnn_test_en_cla
import tensorflow as tf

HEADERS = {
    'Connection': 'keep-alive',
    'Authority': 'www.zhihu.com',
    'Referer': 'https://www.zhihu.com/signup',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'
    }
LOGIN_URL = 'https://www.zhihu.com/signup'
LOGIN_API = 'https://www.zhihu.com/api/v3/oauth/sign_in'
LOGOUT_URL = 'https://www.zhihu.com/logout'
FORM_DATA = {
    'client_id': 'c3cef7c66a1843f8b3a9e6a1e3160e20',
    'grant_type': 'password',
    'source': 'com.zhihu.web',
    'username': 'username',
    'password': 'password',
    # as for Chinese type captcha, change 'en' to 'cn'
    'lang': 'en',
    'ref_source': 'homepage'
}


class ZhihuAccount(object):

    def __init__(self):
        self.login_url = LOGIN_URL
        self.login_api = LOGIN_API
        self.logout_url = LOGOUT_URL
        self.login_data = FORM_DATA.copy()
        self.session = requests.session()
        self.session.headers = HEADERS.copy()
        self.session.cookies = cookiejar.LWPCookieJar(filename='./cookies.txt')
        self.lang = None

    def login_first(self, username=None, password=None, load_cookies=True, lang=None):
        """prepare to login to Zhihu
        :param username:
        :param password:
        :param load_cookies: whether to load cookies stored in last login
        :return: bool
        """
        if load_cookies and self.load_cookies():
            if self.check_login():
                print('登录成功')
                return True

        # 1 get value _sxrf and x-uuid, and update headers using those two values
        token, uuid = self._get_token()
        self.session.headers.update({
            # authorization is a constant value
            'authorization': 'oauth c3cef7c66a1843f8b3a9e6a1e3160e20',
            'X-Xsrftoken': token,
            'x-uuid': uuid
        })
        # update username and password
        username, password = self._check_user_pass(username, password)
        self.login_data.update({
            'username': username,
            'password': password
        })
        self.lang = self.tooglelang(lang)
        print(self.lang)
        return False

    def login(self):
        '''sumilate login to Zhihu
        '''
        timestamp = str(int(time.time()*1000))
        self.login_data.update({
            # get captcha, step 2, 3, 4
            'captcha': self._get_captcha(self.lang),
            'timestamp': timestamp,
            # get signature, if this value is not correct , it will cause security error.
            # Zhihu will send a message to your phone and ask you to change password
            'signature': self._get_signature(timestamp)
        })

        resp = self.session.post(self.login_api, data=self.login_data)
        if 'error' in resp.text:
            print('error:', re.findall(r'"message":"(.+?)"', resp.text)[0])
        elif self.check_login():
            print('登录成功')
            return True
        print('登录失败')
        return False

    def logout(self):
        resp = self.session.get(self.logout_url, headers={'authorization': None, 'X-Xsrftoken': None, 'x-udid': None})
        # print(resp)

    def grep_name(self):
        '''find username and headline after login success
        '''
        if not self.check_login():
            print('Not login')
            return
        resp = self.session.get('https://www.zhihu.com')
        soup = BeautifulSoup(resp.content, 'html5lib')
        with open('resp.html', 'wb') as f:
            f.write(soup.prettify().encode('utf-8'))
        name = re.findall(r'data-state=.*?("|&quot;)name("|&quot;):("|&quot;)(.+?)("|&quot;),("|&quot;)headline("|&quot;):("|&quot;)(.+?)("|&quot;)', soup.prettify())[0]
        print(name[3], name[8])

    def load_cookies(self):
        """load cookies and store it to file
        :return: boolean
        """
        try:
            self.session.cookies.load(ignore_discard=True)
            return True
        except FileNotFoundError:
            return False

    def check_login(self):
        """check login status
        :return: bool
        """
        resp = self.session.get(self.login_url, allow_redirects=False)
        if resp.status_code == 302:
            self.session.cookies.save()
            return True
        return False

    def _get_token(self):
        """get value of _sxrf and x-uuid
        :return:
        """
        resp = self.session.get(self.login_url, headers={'authorization': None, 'X-Xsrftoken': None, 'x-udid': None})
        token = re.findall(r'_xsrf=([\w|-]+)', str(self.session.cookies))[0]
        uuid = re.findall(r'd_c0="([0-9a-zA-Z-_]+?=)', str(self.session.cookies))[0]
        return token, uuid

    def _get_captcha(self, lang='en'):
        """request captcha
        :return: captcha text
        """
        # requests supported_contries before requests captcha, just imitate what broswer does but this step is not necessary, maybe..
        # resp = self.session.get('https://www.zhihu.com/api/v3/oauth/sms/supported_countries', headers={'X-Xsrftoken': None})
        if lang == 'cn':
            api = 'https://www.zhihu.com/api/v3/oauth/captcha?lang=cn'
        else:
            api = 'https://www.zhihu.com/api/v3/oauth/captcha?lang=en'
        # 2 check whether it need to get captcha
        resp = self.session.get(api, headers={'X-Xsrftoken': None})
        show_captcha = re.search(r'true', resp.text)
        print('Need captcha?,', show_captcha != None)
        if show_captcha:
            g = tf.Graph()
            if lang == 'cn':
                with g.as_default():
                    cnntest_cn = cnn_test_cn.CnnTest('result/cn1')
            else:
                with g.as_default():
                    cnntest_cla = cnn_test_en_cla.CnnTest('result/en3_3')
            try_num = 0
            while True:
                # 3 download captcha picture
                resp = self.session.put(api)
                img_base64 = re.findall(
                    r'"img_base64":"(.+)"', resp.text, re.S)[0].replace(r'\n', '')
                with open('./captcha.jpg', 'wb') as f:
                    f.write(base64.b64decode(img_base64))
                img = Image.open('./captcha.jpg')
                img.show()
                if lang == 'cn':
                    capt, num_flip_list = cnntest_cn.cnn_test_single(img)
                    print('num_flip_list:', num_flip_list)
                else:
                    type = cnntest_cla.cnn_test_single(img)
                    if type == 0:
                        g1 = tf.Graph()
                        with g1.as_default():
                            cnntest_en = cnn_test_en.CnnTest('result/en1_3_log')
                    else:
                        g1 = tf.Graph()
                        with g1.as_default():
                            cnntest_en = cnn_test_en.CnnTest('result/en2_1')
                    capt = cnntest_en.cnn_test_single(img)

                print('Recognized Captcha, ' + str(try_num + 1) + '(times) : ' + capt)
                input('<...Pause Input...>')
                # 4 post captcha text to confirm
                resp = self.session.post(api, data={'input_text': capt})
                if resp.status_code == 201:
                    return capt
                if try_num > 2:
                    break
                try_num +=1
        return ''

    def _get_signature(self, timestamp):
        """calculate signature
        :param timestamp:
        :return:
        """
        ha = hmac.new(b'd1b964811afb40118a12068ff74a12f4', digestmod=hashlib.sha1)
        grant_type = self.login_data['grant_type']
        client_id = self.login_data['client_id']
        source = self.login_data['source']
        ha.update(bytes((grant_type + client_id + source + timestamp), 'utf-8'))
        return ha.hexdigest()

    def _check_user_pass(self, username, password):
        """check the username and password
        """
        if username is None:
            username = self.login_data.get('username')
            if not username:
                username = input('请输入手机号：')
        # if '+86' not in username:
        #     username = '+86' + username

        if password is None:
            password = self.login_data.get('password')
            if not password:
                password = input('请输入密码：')
        return username, password

    def tooglelang(self, lang=None):
        '''对同一个IP地址来说，登录知乎的相邻会话(session)会交替切换中文和英文的验证码形式。
        相邻的会话在请求验证码时使用同一种api也能登录知乎，只是第二次没有验证码而已
        由于本实验想要演示的过程是通过自动识别验证码登录知乎，所以会随着会话的切换而改变验证码api形式，
        确保每次登录都需要使用到验证码。
        '''
        if lang is None:
            lang = 'en'
            try:
                with open('setting.txt', 'r', encoding='utf-8') as f:
                    lang = f.read()
            except:
                pass
            if lang == 'en':
                lang = 'cn'
            else:
                lang = 'en'
        with open('setting.txt', 'w', encoding='utf-8') as f:
            f.write(lang)
        return lang

if __name__ == '__main__':
    account = ZhihuAccount()
    lang = account.login_first(load_cookies=False, lang=None)
    login_time = 0
    if account.login():
        account.grep_name()
        # account.logout()
