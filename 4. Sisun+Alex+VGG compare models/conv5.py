import numpy as np
import tensorflow as tf
import os
import time

start = time.time()
logdir = './logs/conv5/' + str(time.time())

np.random.seed(int(time.time()))
tf.set_random_seed(777)  # reproducibility

# hyper parameters
inputDataType = "Conv5_YCbCrR"
learning_rate = 0.0009
# training_epochs = 300 (cost 수렴 epoch)
training_epochs = 300
batch_size = 4000
image_depth = 3

# (전체 데이터 대비 (학습/검증)입력 데이터 비율)
train_data_rate = 0.9
test_data_rate = 0.1

# lrn(2, 2e-05, 0.75, name='norm1')
radius = 2
alpha = 2e-05
beta = 0.75
bias = 1.0

# data input
input_data = 'ycbcrrAll.csv'
# input_data = 'ycbrAll.csv'
data_xy = np.loadtxt(input_data, delimiter=',', dtype=np.float32)
np.random.shuffle(data_xy)
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
# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# define correct_prediction & accuracy
prediction = tf.argmax(logits, 1)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.98)
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
print('learning_rate / training_epochs:', learning_rate, '/', training_epochs)
print('Learning started...')

train_N = (int)(data_N * train_data_rate)
print('train_N:', train_N, 'input')

Training_Cost = 0
start = time.time()
for epoch in range(training_epochs):
    i = 0
    while i <= train_N:
        if (i + batch_size) > train_N:
            batch_train_x = data_xy[i:(i + (train_N % batch_size)), 0:-1]
            batch_train_y = data_xy[i:(i + (train_N % batch_size)), [-1]]
        else:
            batch_train_x = data_xy[i:(i + batch_size), 0:-1]
            batch_train_y = data_xy[i:(i + batch_size), [-1]]

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

fire_trueCnt = 0
fire_falseCnt = 0
nofire_trueCnt = 0
nofire_falseCnt = 0

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

    predict_output, a, s = sess.run([prediction, accuracy, summary], feed_dict=test_feed_dict)

    writer.add_summary(s, global_step=test_step)
    test_step = test_step + 1

    # y_data: (N,1) = flatten => (N, ) matches predict_output.shape
    for p, y in zip(predict_output, batch_test_y.flatten()):
        if p == int(y):
            if int(y) == 1:
                fire_trueCnt = fire_trueCnt + 1
            else:
                nofire_trueCnt = nofire_trueCnt + 1
        else:
            if int(y) == 1:
                fire_falseCnt = fire_falseCnt + 1
            else:
                nofire_falseCnt = nofire_falseCnt + 1

    print('test batch:', i, 'test accuracy', a)
    test_accuracy_sum += a
    test_accuracy_sum_count += 1

    i += batch_size
A = fire_trueCnt + fire_falseCnt
B = fire_trueCnt + nofire_falseCnt
C = A * B
D = A + B
criterion2 = 2 * float(C / D)
print(fire_trueCnt, fire_falseCnt, nofire_trueCnt, nofire_falseCnt)
print(A, B, C, D)
print('Test_Accuracy:', float(test_accuracy_sum / test_accuracy_sum_count))
print('criterion2:', criterion2)
print('Test Finished!\n')

# ---------------------------------------------------------------------------------------------------
# Summary
print('----------------------------------------------------------------------------------------')
print('Last_Training_Cost:', Training_Cost)
print('Test_Accuracy:', float(test_accuracy_sum / test_accuracy_sum_count))
print('criterion2:', criterion2)
print('train execution time(second):', execution_time)
print('logdir:', logdir)

dirName = inputDataType + "_" + str(
    int(100 * (test_accuracy_sum / test_accuracy_sum_count))) + "_" + str(
    training_epochs) + "_" + str(batch_size) + "_" + str(learning_rate)
os.mkdir(str(dirName))
save_path = saver.save(sess, os.getcwd() + "/" + dirName + "/conv5_" + str(
    test_accuracy_sum / test_accuracy_sum_count) + ".ckpt")
print("\nModel saved in file: %s" % dirName)
sess.close()

# ---------------------------------------------------------------------------------------------------
# file write log
f1 = open(dirName + "/model_log.txt", "a")
f1.write('learning_rate: %f\n' % learning_rate)
f1.write('training_epochs: %d\n' % training_epochs)
f1.write('batch_size: %d\n' % batch_size)
f1.write('Cost: %f\n' % Training_Cost)
f1.write('Accuracy: %f\n' % float(test_accuracy_sum / test_accuracy_sum_count))
f1.write('criterion2: %f\n' % criterion2)
f1.write('fire_trueCnt: %d, ' % fire_trueCnt)
f1.write('fire_falseCnt: %d\n' % fire_falseCnt)
f1.write('nofire_trueCnt: %d, ' % nofire_trueCnt)
f1.write('nofire_falseCnt: %d\n' % nofire_falseCnt)
f1.write("----------------------------------------------------------------------------------------------\n")
f1.close()

f = open("./history.txt", 'a')
f.write('-- conv5 %d depth-----------------------\n' % image_depth)
f.write('learning_rate: %f\n' % learning_rate)
f.write('training_epochs: %d\n' % training_epochs)
f.write('batch_size: %d\n' % batch_size)
f.write('Cost: %f\n' % Training_Cost)
f.write('Accuracy: %f\n' % float(test_accuracy_sum / test_accuracy_sum_count))
f.write('criterion2: %f\n' % criterion2)
f.write('fire_trueCnt: %d, ' % fire_trueCnt)
f.write('fire_falseCnt: %d\n' % fire_falseCnt)
f.write('nofire_trueCnt: %d, ' % nofire_trueCnt)
f.write('nofire_falseCnt: %d\n\n' % nofire_falseCnt)
f.close()
