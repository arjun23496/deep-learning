import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

num_steps = 5000
learning_rate = 0.0002
epsilon = 1e-5

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.random_normal([784, 50]))
b1 = tf.Variable(tf.zeros([50]))

W2 = tf.Variable(tf.random_normal([50, 10]))
b2 = tf.Variable(tf.zeros([10]))

l1_out = tf.nn.softmax(tf.matmul(x, W1) + b1)

# pred = tf.nn.softmax(l1_out)

pred = tf.nn.softmax(tf.matmul(l1_out, W2) + b2)

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred+epsilon), reduction_indices=1))

optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
print "initial accuracy: ", acc

for step in range(num_steps):

	batch_xs, batch_ys = mnist.train.next_batch(100)

	prev1 =  sess.run([W1, b1])
	# prev2 = sess.run([W2, b2])

	out, _ = sess.run([pred, train], { x: batch_xs, y:batch_ys })

	# print "out: ", out[0]
	print "loss: ", sess.run([loss], { x: batch_xs, y:batch_ys })

	now1 = sess.run([W1, b1])
	# now2 = sess.run([W2, b2])

	# print now1[0] - prev1[0]
	# print now1[1] - prev1[1]
	# print now2[0] - prev2[0]

acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})

print "accuracy: ",acc