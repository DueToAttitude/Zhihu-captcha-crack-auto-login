# encoding:utf-8

import requests
from hashlib import md5


class RClient(object):
    # 若快api初始化
    def __init__(self, username, password, soft_id, soft_key):
        self.username = username    # 若快账户名
        self.password = md5(password.encode('utf-8')).hexdigest()   # 密码
        self.soft_id = soft_id      # soft_id
        self.soft_key = soft_key    # soft_key
        self.base_params = {
            'username': self.username,
            'password': self.password,
            'softid': self.soft_id,
            'softkey': self.soft_key,
        }
        self.headers = {
            'Connection': 'Keep-Alive',
            'Expect': '100-continue',
            'User-Agent': 'ben',
        }

    def rk_create(self, im, im_type=3040, timeout=60):
        """识别模块
        :param im: 验证码图片，类型为二进制文件
        :param im_type: 识别类型，3040表示4个英文以及数字字符的验证码形式，更多形式见若快官网：www.ruokuai.com
        :param timeout: 超时时间，尽量设置得大一点，防止超时抛出异常使得整个代码被终止，也可以自己处理异常
        """
        params = {
            'typeid': im_type,
            'timeout': timeout,
        }
        params.update(self.base_params)
        files = {'image': ('a.jpg', im)}
        r = requests.post('http://api.ruokuai.com/create.json', data=params, files=files, headers=self.headers)
        return r.json()

    def rk_report_error(self, im_id):
        """
        im_id:报错题目的ID
        """
        params = {
            'id': im_id,
        }
        params.update(self.base_params)
        r = requests.post('http://api.ruokuai.com/reporterror.json', data=params, headers=self.headers)
        return r.json()


if __name__ == '__main__':
    # 测试
    rc = RClient('username', 'password', 'soft_id', 'soft_key')
    im = open('a.jpg', 'rb').read()
    print(rc.rk_create(im, 3040))

