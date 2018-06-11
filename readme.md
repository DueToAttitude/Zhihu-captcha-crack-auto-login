# 基于CNN的知乎模拟登陆
使用CNN自动识别知乎验证码，包括中文验证码和英文验证码；登录使用selenium库

**关键词**：`CNN`、`tensorflow`、`知乎验证码`、`模拟登陆`

## 1、介绍
知乎验证码分为**中文验证码**和**英文验证码**，其中英文验证码具有两种不同的样式，记为**样式1**和**样式2**

>样式1

![pic][en1]

>样式2

![pic][en2]

>中文验证码

![pic][cn_zhihu]

## 2、思路
### 1) CNN模型
训练4个CNN模型，分别为：

1. **中文识别模型**：识别中文倒立汉字位置
2. **英文样式分类模型**：对英文验证码不同样式进行分类
3. **英文验证码样式1识别模型**：识别样式1的英文数字字符串
4. **英文验证码样式2识别模型**：识别样式2的英文数字字符串

### 2) CNN模型框架

![pic][cnn_graph]

### 3) 样本数据收集
#### 英文验证码 
1. 知乎验证码接口爬取
2. 提交若快打码平台打码
3. 相同session内提交若快识别结果到知乎验证码接口
4. 识别正确，则保存文件，文件名为识别结果

#### 中文验证码
没找到能够识别知乎中文倒立汉字的打码平台，不想人工打码，于是模仿知乎中文验证码样式，使用Pillow库自己生成。

>生成的验证码示例

![pic][cn_draw]


## 3、文件列表
* **result**: 存储训练好的CNN模型
    * **cn**: 中文验证码识别模型
    * **en_cla**: 英文验证码样式分类模型
    * **en1**: 英文验证码样式1识别模型
    * **en2**: 英文验证码样式2识别模型
* **source**: 一些资源文件
    * **chromedriver.exe/MicrosoftWebDriver.exe**: selenium库要用到的WebDriver
    * **font_type**: 生成中文验证码使用的中文字体
    * **captcha_data.rar**: 数据集压缩文件，只上传小部分样本作为参考
        * **cn_draw**: 生成的中文验证码示例
        * **cn_zhihu**: 知乎中文验证码示例
        * **en1**: 英文验证码样式1数据集，文件大小超过100M限制，于是压缩
        * **en2**: 英文验证码样式2数据集，同上，压缩
* **captcha_get.py**: 知乎验证码样本收集
* **captcha_generate.py**: 中文验证码生成
* **cnn_test_*.py**: CNN模型测试
* **cnn_train_*.py**: CNN模型训练
* **login.py**: 模拟登陆知乎，只使用requests库
* **login_sele.py**: 模拟登陆知乎，使用selenium库
* **util_cn.py**: 中文验证码通用函数
* **util_en.py**: 英文验证码通用函数
* **rk.py**: 若快打码api

## 4、如何使用
### 1) 安装依赖
使用Python3，依赖库：

| 依赖库 | 版本 |
| ----------------- | --------- |
| beautifulsoup4    | 4.6.0     |
| numpy             | 1.14.2    |
| Pillow            | 5.1.0     |
| requests          | 2.18.4    |
| scipy             | 1.0.1     |
| selenium          | 3.12.0    |
| tensorflow-gpu    | 1.8.0     |

### 2)安装中文字体
字体文件位于source/font_type文件夹中，也可以安装更加适合的字体，可自行选择

### 3) 将WebDriver放置在系统Path路径下
WebDriver位于source文件夹中

### 4) 模拟登陆
result中已经有训练完成的CNN模型，可直接用于模拟登陆，运行login_sele.py即可

### 5) 训练模型
也可以自己训练CNN模型，文件列表中已有数据集，自己构造训练集测试集进行CNN模型训练，运行相应的模型训练文件cnn_train_*.py

### 6) 其他
* 需要安装Chrome浏览器，尽量将Chrome浏览器更新至最新版本
* 可用通过"pip install -r requirements.txt"命令安装依赖库，前提是已经安装pip
* 通过此方法安装的tensorflow是GPU版本，GPU版本的tensorflow需要另外安装依赖软件，请自行解决

## 5、To-do-list


--------------------------------
[en1]:/source/readme_img/en1.gif
[en2]:/source/readme_img/en2.gif
[cn_zhihu]:/source/readme_img/cn_zhihu.gif
[cn_draw]:/source/readme_img/cn_draw.png
[cnn_graph]:/source/readme_img/cnn_graph.png