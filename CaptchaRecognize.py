#coding=utf-8

import tensorflow as tf
import numpy as np
from PIL import Image
from skimage import io
import os
import random
import matplotlib.pyplot as plt
from  PicUtil import  *
import cv2 as cv
import  urllib2,urllib
import time

######这里处理的图片都进行了二值化处理

#训练图片路径
train_data_dir = r'/train/picdir/'

#测试图片路径
test_data_dir = r'/test/picdir/'

#模型存放路径
modelPath='/model'

#图片字符统一高
img_height=50
#图片字符统一宽
img_width=50
#图片里面的字符数
max_captcha = 1
#待识别字符个数
char_len =36

###初始化待识别字符,这个根据验证码特征可调整
charIndex=['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
# 获取批量训练的字符
def get_train_data(batch_size=32):
    '''
    生成训练数据
    :param batch_size: 每次训练载入的图片得数目，默认为32
    '''
    train_file_name_list = os.listdir(train_data_dir)
    selected_train_file_name_list = random.sample(train_file_name_list, batch_size)
    batch_x = np.zeros([batch_size, img_height*img_width])
    batch_label = np.zeros([batch_size, max_captcha*char_len])
    i=0
    for selected_train_file_name in selected_train_file_name_list:
        if selected_train_file_name.endswith('.png'):
            captcha_image = Image.open(os.path.join(train_data_dir, selected_train_file_name))
            captcha_image_np = np.array(captcha_image)
            assert captcha_image_np.shape == (img_height, img_width)
            captcha_image_np =captcha_image_np.flatten()/255
            print(captcha_image_np)
            batch_x[i,:]=captcha_image_np
            y_temp_value=np.array(list(selected_train_file_name[0]))
            print(y_temp_value)
            tempindex=charIndex.index(y_temp_value)
            outSample= np.zeros(char_len,dtype=np.int)
            outSample[tempindex]=1
            batch_label[i, :] = outSample
            i=i+1

    return batch_x, batch_label

# 初始化权值
def weight_variable(shape, name='weight'):
    init = tf.truncated_normal(shape, stddev=0.1)
    var = tf.Variable(initial_value=init, name=name)
    return var

# 初始化偏置
def bias_variable(shape, name='bias'):
    init = tf.constant(0.1, shape=shape)
    var = tf.Variable(init, name=name)
    return var

# 卷积
def conv2d(x, W, name='conv2d'):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)

# 池化
def max_pool_2X2(x, name='maxpool'):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

#构建卷积神经网络
def cnn_graph(X,keep_prob):
        # 输入层
    # 请注意 X 的 name，在测试model时会用到它
    x_input = tf.reshape(X, [-1, img_height, img_width, 1], name='x-input')
    # dropout,防止过拟合
    # 第一层卷积
    W_conv1 = weight_variable([5, 5, 1, 32], 'W_conv1')
    B_conv1 = bias_variable([32], 'B_conv1')
    conv1 = tf.nn.relu(conv2d(x_input, W_conv1, 'conv1') + B_conv1)
    conv1 = max_pool_2X2(conv1, 'conv1-pool')
    #扔掉部分神经元
    conv1 = tf.nn.dropout(conv1, keep_prob)
    # 第二层卷积
    W_conv2 = weight_variable([5, 5, 32, 64], 'W_conv2')
    B_conv2 = bias_variable([64], 'B_conv2')
    conv2 = tf.nn.relu(conv2d(conv1, W_conv2, 'conv2') + B_conv2)
    conv2 = max_pool_2X2(conv2, 'conv2-pool')
    conv2 = tf.nn.dropout(conv2, keep_prob)
    # 第三层卷积
    W_conv3 = weight_variable([5, 5, 64, 64], 'W_conv3')
    B_conv3 = bias_variable([64], 'B_conv3')
    conv3 = tf.nn.relu(conv2d(conv2, W_conv3, 'conv3') + B_conv3)
    conv3 = max_pool_2X2(conv3, 'conv3-pool')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    t_img_height=int(conv3.shape[1])
    t_img_width=int(conv3.shape[2])
    # 全链接层
    # 每次池化后，图片的宽度和高度均缩小为原来的一半，进过上面的三次池化，宽度和高度均缩小8倍
    W_fc1 = weight_variable([t_img_height * t_img_width * 64, 1024], 'W_fc1')
    B_fc1 = bias_variable([1024], 'B_fc1')
    fc1 = tf.reshape(conv3, [-1, t_img_height * t_img_width * 64])
    fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, W_fc1), B_fc1))
    fc1 = tf.nn.dropout(fc1, keep_prob)
    # 输出层
    W_fc2 = weight_variable([1024, max_captcha * char_len], 'W_fc2')
    B_fc2 = bias_variable([max_captcha * char_len], 'B_fc2')
    output = tf.add(tf.matmul(fc1, W_fc2), B_fc2, 'output')
    return  output

# 优化参数
def optimize_graph(output,Y):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
    return optimizer


def accuracy_graph(output,Y):
    predict = tf.reshape(output, [-1, max_captcha, char_len], name='predict')
    labels = tf.reshape(Y, [-1, max_captcha, char_len], name='labels')
    # 预测结果
    # 请注意 predict_max_idx 的 name，在测试model时会用到它
    predict_max_idx = tf.argmax(predict, axis=2, name='predict_max_idx')
    labels_max_idx = tf.argmax(labels, axis=2, name='labels_max_idx')
    predict_correct_vec = tf.equal(predict_max_idx, labels_max_idx)
    accuracy = tf.reduce_mean(tf.cast(predict_correct_vec, tf.float32))
    return  accuracy

# 训练
def train():
    print("start train")
    X = tf.placeholder(tf.float32, [None, img_width * img_height], name='data-input')
    Y = tf.placeholder(tf.float32, [None, max_captcha * char_len], name='label-input')
    # 请注意 keep_prob 的 name，在测试model时会用到它
    keep_prob = tf.placeholder(tf.float32, name='keep-prob')
    # cnn模型
    y_conv = cnn_graph(X,keep_prob)
    # 优化
    optimizer = optimize_graph(y_conv,Y)
    # 计算准确率
    accuracy = accuracy_graph(y_conv,Y)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        steps = 0
        for epoch in range(6000):
            print(epoch)
            train_data, train_label = get_train_data(64)
            sess.run(optimizer, feed_dict={X: train_data, Y: train_label, keep_prob: 0.75})
            if steps % 100 == 0:
                test_data, test_label = get_train_data()
                acc = sess.run(accuracy, feed_dict={X: test_data, Y: test_label, keep_prob: 1.0})
                print("steps=%d, accuracy=%f" % (steps, acc))
                if acc > 0.99:
                    saver.save(sess,  "./crack_captcha.model", global_step=steps)
                    break
            steps += 1
    print("train over")

# 识别
def recognizeCaptcha(picPath):
    ##加载处理好的图片
    picList=img=io.imread(picPath)
    X = tf.placeholder(tf.float32, [None, img_width * img_height], name='data-input')
    Y = tf.placeholder(tf.float32, [None, max_captcha * char_len], name='label-input')
    # 请注意 keep_prob 的 name，在测试model时会用到它
    keep_prob = tf.placeholder(tf.float32, name='keep-prob')
    y_conv = cnn_graph(X, keep_prob)
    captchaValue=''
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(modelPath))
        picNumber = len(picList)
        for num in range(picNumber):
            img_data = picList[num]
            captcha_image_np = np.array(img_data)
            # 在什么值后面插入值
            captcha_image_np = captcha_image_np.flatten() / 255

            predict_max_idx = tf.argmax(tf.reshape(y_conv, [-1, max_captcha, char_len]), 2)
            predict = sess.run(predict_max_idx, feed_dict={X: [captcha_image_np], keep_prob: 1.0})
            predictValue = np.squeeze(predict)
            recognizeValue = charIndex[predictValue]
            captchaValue=captchaValue+recognizeValue
    return captchaValue



