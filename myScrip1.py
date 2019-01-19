import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs, keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys,1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(acc, feed_dict={xs:v_xs, ys:v_ys, keep_prob:1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

xs = tf.placeholder(tf.float32,[None, 784])/255.
ys = tf.placeholder(tf.float32,[None, 10])
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(xs,[-1,28,28,1])

w_l1 = weight_variable([5,5,1,32])
b_l1 = bias_variable([32])
h_l1 = tf.nn.relu(conv2d(x_image,w_l1)+b_l1)
p_l1 = max_pool_2x2(h_l1)

w_l2 = weight_variable([5,5,32,64])
b_l2 = bias_variable([64])
h_l2 = tf.nn.relu(conv2d(p_l1,w_l2)+b_l2)
p_l2 = max_pool_2x2(h_l2)

p_l2_plat = tf.reshape(p_l2,[-1,7*7*64])
w_f1 = weight_variable([7*7*64, 1024])
b_f1 = bias_variable([1024])
h_f1 = tf.nn.relu(tf.matmul(p_l2_plat, w_f1)+b_f1)
h_f1_dr = tf.nn.dropout(h_f1, keep_prob)

w_f2 = weight_variable([1024, 10])
b_f2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_f1_dr, w_f2)+b_f2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                              reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys, keep_prob: 0.5})
    if i %50 == 0:
        print(compute_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000]))
        













