import numpy as np
import tensorflow as tf
import random
import os
import time

start = time.time()
logdir = './logs/new/' + str(time.time())

np.random.seed(int(time.time()))
tf.set_random_seed(777)  # reproducibility

# hyper parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 300
image_depth = 2

# (전체 데이터 대비 (학습/검증)입력 데이터 비율)
train_data_rate = 0.8
test_data_rate = 0.2

# Filter 크기(Filter Size = FS)
FS = 4
# Filter 개수(Filter Num = FN)
FN = 32

# lrn(2, 2e-05, 0.75, name='norm1')
radius = 2
alpha = 2e-05
beta = 0.75
bias = 1.0

# data input
input_data = '32DATA.csv'
data_xy = np.loadtxt(input_data, delimiter=',', dtype=np.float32)
np.random.shuffle(data_xy)
data_N = len(data_xy) # print('data_N: ', data_N)

# ---------------------------------------------------------------------------------------------------
# X: input 32*32*image_depth
# Y: output '1' or '0'
X = tf.placeholder(tf.float32, [None, 32*32*image_depth])
Y = tf.placeholder(tf.int32, [None, 1])  # 0,1

# 출력 class 개수 = 1(fire), 0(not fire)
nb_classes = 2

# one hot & reshape
Y_one_hot = tf.one_hot(Y, nb_classes) # print("one_hot", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) # print("reshape", Y_one_hot)

# img 32x32ximage_depth (black/white)
X_img = tf.reshape(X, [-1, 32, 32, image_depth])

# ---------------------------------------------------------------------------------------------------
# L1 ImgIn shape = (?, 32, 32, image_depth)
W1 = tf.Variable(tf.random_normal([FS, FS, image_depth, FN], stddev=0.01))

# Conv -> (?, 32, 32, FN)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)

# lrn1
# lrn(2, 2e-05, 0.75, name='norm1')
L1 = tf.nn.local_response_normalization(L1, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

# Pool -> (?, 16, 16, FN)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# ---------------------------------------------------------------------------------------------------
# L2 ImgIn shape = (?, 16, 16, FN)
W2 = tf.Variable(tf.random_normal([FS, FS, FN, FN], stddev=0.01))

# Conv -> (?, 16, 16, FN)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)

# lrn2
# lrn(2, 2e-05, 0.75, name='norm1')
L2 = tf.nn.local_response_normalization(L2, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

# Pool -> (?, 8, 8, FN)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# ---------------------------------------------------------------------------------------------------
# L3 ImgIn shape = (?, 8, 8, FN)
W3 = tf.Variable(tf.random_normal([FS, FS, FN, FN], stddev=0.01))

# Conv -> (?, 8, 8, FN)
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)

# ---------------------------------------------------------------------------------------------------
# L4 ImgIn shape = (?, 8, 8, FN)
W4 = tf.Variable(tf.random_normal([FS, FS, FN, FN], stddev=0.01))

# Conv -> (?, 8, 8, FN)
L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
L4 = tf.nn.relu(L4)

# ---------------------------------------------------------------------------------------------------
# L5 ImgIn shape = (?, 8, 8, FN)
W5 = tf.Variable(tf.random_normal([FS, FS, FN, FN], stddev=0.01))

# Conv -> (?, 8, 8, FN)
L5 = tf.nn.conv2d(L4, W5, strides=[1, 1, 1, 1], padding='SAME')
L5 = tf.nn.relu(L5)

# Pool -> (?, 4, 4, FN)
L5 = tf.nn.max_pool(L5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Reshape -> (?, 4 * 4 * FN) - Flatten them for FC
L5_flat = tf.reshape(L5, [-1, 4 * 4 * FN])

# ---------------------------------------------------------------------------------------------------
# L6 FC 4x4xFN inputs ->  400 outputs
W6 = tf.get_variable("W6", shape=[FN * 4 * 4, 400], initializer = tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([400]))
L6 = tf.nn.relu(tf.matmul(L5_flat, W6) + b6)

# ---------------------------------------------------------------------------------------------------
# L7 FC 4096 inputs ->  1000 outputs
W7 = tf.get_variable("W7", shape=[400, 400], initializer = tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.random_normal([400]))
L7 = tf.nn.relu(tf.matmul(L6, W7) + b7)

# ---------------------------------------------------------------------------------------------------
# L8 FC 1000 inputs -> 1 outputs
W8 = tf.get_variable("W8", shape=[400, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
b8 = tf.Variable(tf.random_normal([nb_classes]))
logits = tf.matmul(L7, W8) + b8

# ---------------------------------------------------------------------------------------------------
# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# define correct_prediction & accuracy
prediction = tf.argmax(logits, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
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
writer = tf.summary.FileWriter(logdir)
writer.add_graph(sess.graph)

# Add ops to save and restore all the variables. # ★
saver = tf.train.Saver()

# ---------------------------------------------------------------------------------------------------
# train my model and check cost
print('Learning started...')

train_N = (int)(data_N * train_data_rate)
print('train_N:', train_N, 'input')

Training_Cost = 0
start = time.time()
for epoch in range(training_epochs):
    i = 0
    while i <= train_N:
        if (i+batch_size) > train_N:
            batch_train_x = data_xy[i:(i + (train_N % batch_size)), 0:-1]
            batch_train_y = data_xy[i:(i + (train_N % batch_size)), [-1]]
        else:
            batch_train_x = data_xy[i:(i+batch_size), 0:-1]
            batch_train_y = data_xy[i:(i+batch_size), [-1]]

        train_feed_dict = {X: batch_train_x, Y: batch_train_y}

        Training_Cost, _ = sess.run([cost, optimizer], feed_dict=train_feed_dict)

        print('train epoch:', epoch, 'batch:', i, 'cost:', Training_Cost)

        i += batch_size

print('Last_Training_Cost:', Training_Cost)
print('Learning Finished!\n')
execution_time = time.time() - start
# ---------------------------------------------------------------------------------------------------
# Test model and check accuracy
print('Test started...')

test_N = (int)(data_N * test_data_rate)
print('test_N:', test_N, 'input')

test_accuracy_sum = 0
test_accuracy_sum_count = 0
test_step = test_N
i = train_N
while i <= (train_N + test_N):
    if (i + batch_size) > test_N:
        batch_test_x = data_xy[i:(i + (test_N % batch_size)), 0:-1]
        batch_test_y = data_xy[i:(i + (test_N % batch_size)), [-1]]
    else:
        batch_test_x = data_xy[i:(i + batch_size), 0:-1]
        batch_test_y = data_xy[i:(i + batch_size), [-1]]

    test_feed_dict = {X: batch_test_x, Y: batch_test_y}

    a, s = sess.run([accuracy, summary], feed_dict=test_feed_dict)

    writer.add_summary(s, global_step=test_step)
    test_step = test_step + 1

    print('test batch:', i, 'test accuracy', a)
    test_accuracy_sum += a
    test_accuracy_sum_count += 1

    i += batch_size

print('Test_Accuracy:', float(test_accuracy_sum / test_accuracy_sum_count))
print('Test Finished!\n')

# ---------------------------------------------------------------------------------------------------
# Summary
print('----------------------------------------------------------------------------------------')
print('Last_Training_Cost:', Training_Cost)
print('Test_Accuracy:', float(test_accuracy_sum / test_accuracy_sum_count))
print('execution time(second):', execution_time)
print('logdir:', logdir)

# ---------------------------------------------------------------------------------------------------
# Save the variables to disk.
save_path = saver.save(sess, os.getcwd()+"/new.ckpt")
print("\nModel saved in file: %s" % save_path)

sess.close()

f = open("./history.txt", 'a')
f.write('-- new %d depth-----------------------\n' % image_depth)
f.write('learning_rate: %.5f\n' % learning_rate)
f.write('training_epochs: %d\n' % training_epochs)
f.write('batch_size: %d\n' % batch_size)
f.write('train/test: %.3f / %.3f\n' % (train_data_rate, test_data_rate))
f.write('FS/FN: %d / %d\n' % (FS, FN))
f.write('Cost: %f\n' % Training_Cost)
f.write('Acc: %f\n\n' % float(test_accuracy_sum / test_accuracy_sum_count))
f.close()
