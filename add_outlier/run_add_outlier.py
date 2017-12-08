import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from mnist_deep import deepnn
from minist_with_outlier import MnistWithOutlier


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--outlier_nums', nargs='+', type=int)
    parser.add_argument('--no_outlier_nums', nargs='+', type=int)
    parser.add_argument('--outlier_ratio', type=float)

    args = parser.parse_args()

    # load mnist
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # generate mnist with outlier
    mnist_outlier = MnistWithOutlier(mnist, args.outlier_nums, args.no_outlier_nums, args.outlier_ratio)

    # some paramater
    num_class = mnist_outlier.train_labels.shape[1]

    # train
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.int32, [None, num_class])
    y_conv, keep_prob = deepnn(x, num_class)
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)
    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
          batch = mnist_outlier.next_batch(50)
          if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
          train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist_outlier.test_images, y: mnist_outlier.test_labels, keep_prob: 1.0}))

if __name__ == '__main__':
    main()