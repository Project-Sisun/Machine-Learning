# <화재 감지 모델>
# input: img 32x32x1 (black/white)
# Layer 7개
# Training epochs 1
# Filter 크기 1*1, Filter 개수 16, Filter strides 1*1
# No pooling
# No dropout
# No ensemble

import numpy as np
import tensorflow as tf
import random
import os
import time

start = time.time()
np.random.seed(int(time.time()))
tf.set_random_seed(777)  # reproducibility

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 300

# (전체 데이터 대비 (학습/검증)입력 데이터 비율)
train_data_rate = 0.6
test_data_rate = 0.4

# Filter 크기(Filter Size = FS)
FS = 4
# Filter 개수(Filter Num = FN)
FN = 48

# data input
input_data = 'GrayInputData.csv'
data_xy = np.loadtxt(input_data, delimiter=',', dtype=np.float32)
np.random.shuffle(data_xy)
data_N = len(data_xy) # print('data_N: ', data_N)
# ---------------------------------------------------------------------------------------------------
# X: input 32*32*1 (=1024)
# Y: output '1' or '0'
X = tf.placeholder(tf.float32, [None, 1024])
Y = tf.placeholder(tf.int32, [None, 1])  # 0,1

# 출력 class 개수 = 1(fire), 0(not fire)
nb_classes = 2

# one hot & reshape
Y_one_hot = tf.one_hot(Y, nb_classes) # print("one_hot", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) # print("reshape", Y_one_hot)

# img 32x32x1 (black/white)
X_img = tf.reshape(X, [-1, 32, 32, 1])

# ---------------------------------------------------------------------------------------------------
# L1 ImgIn shape = (?, 32, 32, 1)
W1 = tf.Variable(tf.random_normal([FS, FS, 1, FN], stddev=0.01))

# Conv -> (?, 32, 32, FN)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)

# ---------------------------------------------------------------------------------------------------
# L2 ImgIn shape = (?, 32, 32, FN)
W2 = tf.Variable(tf.random_normal([FS, FS, FN, FN], stddev=0.01))

# Conv -> (?, 32, 32, FN)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)

# ---------------------------------------------------------------------------------------------------
# L3 ImgIn shape = (?, 32, 32, FN)
W3 = tf.Variable(tf.random_normal([FS, FS, FN, FN], stddev=0.01))

# Conv -> (?, 32, 32, FN)
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)

# ---------------------------------------------------------------------------------------------------
# L4 ImgIn shape = (?, 32, 32, FN)
W4 = tf.Variable(tf.random_normal([FS, FS, FN, FN], stddev=0.01))

# Conv -> (?, 32, 32, FN)
L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
L4 = tf.nn.relu(L4)

# ---------------------------------------------------------------------------------------------------
# L5 ImgIn shape = (?, 32, 32, FN)
W5 = tf.Variable(tf.random_normal([FS, FS, FN, FN], stddev=0.01))

# Conv -> (?, 32, 32, FN)
L5 = tf.nn.conv2d(L4, W5, strides=[1, 1, 1, 1], padding='SAME')
L5 = tf.nn.relu(L5)

# ---------------------------------------------------------------------------------------------------
# L6 ImgIn shape = (?, 32, 32, FN)
W6 = tf.Variable(tf.random_normal([FS, FS, FN, FN], stddev=0.01))

# Conv -> (?, 32, 32, FN)
L6 = tf.nn.conv2d(L5, W6, strides=[1, 1, 1, 1], padding='SAME')
L6 = tf.nn.relu(L6)

# ---------------------------------------------------------------------------------------------------
# L7 ImgIn shape = (?, 32, 32, FN)
W7 = tf.Variable(tf.random_normal([FS, FS, FN, FN], stddev=0.01))

# Conv -> (?, 32, 32, FN)
L7 = tf.nn.conv2d(L6, W7, strides=[1, 1, 1, 1], padding='SAME')
L7 = tf.nn.relu(L7)

# Reshape -> (?, 32 * 32 * FN) - Flatten them for FC
L7_flat = tf.reshape(L7, [-1, 32 * 32 * FN])

# ---------------------------------------------------------------------------------------------------
# L8 FC!!! 32x32xFN inputs -> 1 outputs
W8 = tf.get_variable("W8", shape=[32 * 32 * FN, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
b8 = tf.Variable(tf.random_normal([nb_classes]))
logits = tf.matmul(L7_flat, W8) + b8

# ---------------------------------------------------------------------------------------------------
# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# define correct_prediction & accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

# Summary
cost_sum = tf.summary.scalar("cost", cost)
accuracy_sum = tf.summary.scalar("accuracy", accuracy)
summary = tf.summary.merge_all()
# Create summary writer
writer = tf.summary.FileWriter('./logs/7Layer')
writer.add_graph(sess.graph)

# ---------------------------------------------------------------------------------------------------
# train my model and check cost
print('Learning started...')

train_cost_sum = 0
train_N = (int)(data_N * train_data_rate)
print('train_N:', train_N, 'input')

train_step = train_N

for epoch in range(training_epochs):
    i = 0
    while i <= train_N:
        batch_train_x = data_xy[i:(i+batch_size), 0:-1] # [ 109.  104.   84. ...,   83.  109.   90.] (batch_size, 1024)
        batch_train_y = data_xy[i:(i+batch_size), [-1]] # 0.0 (batch_size, 1)

        train_feed_dict = {X: batch_train_x, Y: batch_train_y}

        c, s, _ = sess.run([cost, summary, optimizer], feed_dict=train_feed_dict)
        writer.add_summary(s, global_step=train_step)
        train_step = train_step + 1

        print('train epoch:', epoch, 'batch:', i, 'cost:', c)
        train_cost_sum += c

        i += batch_size

print('Training_Cost:', train_cost_sum / float((train_N / batch_size) * training_epochs))
print('Learning Finished!\n')

# ---------------------------------------------------------------------------------------------------
# Test model and check accuracy
print('Test started...')

test_accuracy_sum = 0
test_N = (int)(data_N * test_data_rate)
print('test_N:', test_N, 'input')

i = train_N
while i <= (train_N + test_N):
    batch_test_x = data_xy[i:(i+batch_size), 0:-1]
    batch_test_y = data_xy[i:(i+batch_size), [-1]]

    test_feed_dict = {X: batch_test_x, Y: batch_test_y}

    a = sess.run(accuracy, feed_dict=test_feed_dict)
    print('test batch:', i, 'test accuracy', a)
    test_accuracy_sum += a

    i += batch_size

print('Test_Accuracy:', test_accuracy_sum / float(test_N / batch_size))
print('Test Finished!\n')

sess.close()

# ---------------------------------------------------------------------------------------------------
# Summary
print('----------------------------------------------------------------------------------------')
print('input_data: ', input_data)
print('7 Layer model')
print('Filter size: ', FS, '*', FS)
print('Filter num: ', FN, '개')
print('----------------------------------------------------------------------------------------')
print('Training_Cost:', train_cost_sum / float((train_N / batch_size) * training_epochs))
print('Test_Accuracy:', test_accuracy_sum / float(test_N / batch_size))
print('execution time(second):', time.time() - start)