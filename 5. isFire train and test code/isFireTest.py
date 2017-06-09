import numpy as np
import tensorflow as tf
import random
import os

tf.set_random_seed(777)

ckptName = "/ISFire_epoch9_cost0.682642.ckpt"

# hyper parameters
inputDataType = "YCbCrR"
batch_size = 5000
image_depth = 3

# lrn(2, 2e-05, 0.75, name='norm1')
radius = 2
alpha = 2e-05
beta = 0.75
bias = 1.0

# data input
input_data = 'ycbcrRTest.csv'
data_xy = np.loadtxt(input_data, delimiter=',', dtype=np.float32)
data_N = len(data_xy)  # print('data_N: ', data_N)

# ---------------------------------------------------------------------------------------------------
# X: input 32*32*image_depth
# Y: output '1' or '0'
X = tf.placeholder(tf.float32, [None, 32 * 32 * image_depth])
Y = tf.placeholder(tf.int32, [None, 1])  # 0,1

# 출력 class 개수 = 1(fire), 0(not fire)
nb_classes = 2

# one hot & reshape
Y_one_hot = tf.one_hot(Y, nb_classes)  # print("one_hot", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])  # print("reshape", Y_one_hot)

# img 32x32x1 (black/white)
X_img = tf.reshape(X, [-1, 32, 32, image_depth])

# ---------------------------------------------------------------------------------------------------
# L1 ImgIn shape = (?, 32, 32, image_depth)
W1 = tf.Variable(tf.random_normal([3, 3, image_depth, 64], stddev=0.01))

# Conv1 -> (?, 32, 32, 64)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')

# Conv2 -> (?, 32, 32, 64)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)

# lrn1
# lrn(2, 2e-05, 0.75, name='norm1')
L1 = tf.nn.local_response_normalization(L1, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

# Pool -> (?, 16, 16, 64)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# ---------------------------------------------------------------------------------------------------
# L2 ImgIn shape = (?, 16, 16, 64)
W2 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))

# Conv1 -> (?, 16, 16, 128)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')

# Conv2 -> (?, 16, 16, 128)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)

# lrn2
# lrn(2, 2e-05, 0.75, name='norm1')
L2 = tf.nn.local_response_normalization(L2, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

# Pool -> (?, 8, 8, 128)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# ---------------------------------------------------------------------------------------------------
# L3 ImgIn shape = (?, 8, 8, 128)
W3 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01))

# Conv1 -> (?, 8, 8, 256)
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')

# Conv2 -> (?, 8, 8, 256)
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')

# Conv3 -> (?, 8, 8, 256)
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)

# Pool -> (?, 4, 4, 256)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# ---------------------------------------------------------------------------------------------------
# L4 ImgIn shape = (?, 4, 4, 256)
W4 = tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=0.01))

# Conv1 -> (?, 4, 4, 512)
L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')

# Conv2 -> (?, 4, 4, 512)
L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')

# Conv3 -> (?, 4, 4, 512)
L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
L4 = tf.nn.relu(L4)

# Pool -> (?, 2, 2, 512)
L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# ---------------------------------------------------------------------------------------------------
# L5 ImgIn shape = (?, 2, 2, 512)
W5 = tf.Variable(tf.random_normal([2, 2, 512, 512], stddev=0.01))

# Conv1 -> (?, 2, 2, 512)
L5 = tf.nn.conv2d(L4, W5, strides=[1, 1, 1, 1], padding='SAME')

# Conv2 -> (?, 2, 2, 512)
L5 = tf.nn.conv2d(L4, W5, strides=[1, 1, 1, 1], padding='SAME')

# Conv3 -> (?, 2, 2, 512)
L5 = tf.nn.conv2d(L4, W5, strides=[1, 1, 1, 1], padding='SAME')
L5 = tf.nn.relu(L5)

# Pool -> (?, 1, 1, 512)
L5 = tf.nn.max_pool(L5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Reshape -> (?, 1 * 1 * 512) - Flatten them for FC
L5_flat = tf.reshape(L5, [-1, 1 * 1 * 512])

# ---------------------------------------------------------------------------------------------------
# L6 FC 1x1x512 inputs ->  4096 outputs
W6 = tf.get_variable("W10", shape=[512 * 1 * 1, 4096], initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([4096]))
L6 = tf.nn.relu(tf.matmul(L5_flat, W6) + b6)

# ---------------------------------------------------------------------------------------------------
# L7 FC 4096 inputs ->  1000 outputs
W7 = tf.get_variable("W7", shape=[4096, 1000], initializer=tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.random_normal([1000]))
L7 = tf.nn.relu(tf.matmul(L6, W7) + b7)

# ---------------------------------------------------------------------------------------------------
# L8 FC 1000 inputs -> 1 outputs
W8 = tf.get_variable("W8", shape=[1000, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
b8 = tf.Variable(tf.random_normal([nb_classes]))
logits = tf.matmul(L7, W8) + b8

# ---------------------------------------------------------------------------------------------------
# define correct_prediction & accuracy
prediction = tf.argmax(logits, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

# Add ops to save and restore all the variables. # ★
saver = tf.train.Saver()
saver.restore(sess, os.getcwd() + ckptName)

# ---------------------------------------------------------------------------------------------------
trueCnt = 0
falseCnt = 0
test_count = 0
test_N = data_N
print('test_N:', test_N, 'input')
fire_falseCnt = 0
nofire_falseCnt = 0
i = 0
while i <= test_N:
    batch_test_x = data_xy[i:(i+batch_size), 0:-1]
    batch_test_y = data_xy[i:(i+batch_size), [-1]]

    predict_output = sess.run(prediction, feed_dict={X: batch_test_x})

    # y_data: (N,1) = flatten => (N, ) matches predict_output.shape
    for p, y in zip(predict_output, batch_test_y.flatten()):
        if p == int(y):
            trueCnt = trueCnt + 1
        else:
            falseCnt = falseCnt + 1
            if int(y) == 1:
                fire_falseCnt = fire_falseCnt + 1
            else:
                nofire_falseCnt = nofire_falseCnt + 1

        test_count = test_count + 1

        # print(i, "[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))

    i += batch_size
    print('batch: %d test finished' % i)


print("count TRUE: ", trueCnt)
print("count FALSE: ", falseCnt)
print('Test_Accuracy:', float(trueCnt / test_count))
sess.close()

f = open("./model_log.txt", 'a')
f.write('\nckpt: %s\n' % ckptName)
f.write('Accuracy: %f\n' % float(trueCnt / test_count))
f.write('fire miss classification: %d\n' % fire_falseCnt)
f.write('nofire miss classification: %d\n' % nofire_falseCnt)
f.close()
