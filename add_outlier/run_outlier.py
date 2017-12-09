import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from mnist_outlier import MnistOutlier


def variable_summaries(self, var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def deepnn(x, num_class):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 10)

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, num_class])
    b_fc2 = bias_variable([num_class])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('steps', type=int)
    parser.add_argument('outlier_ratio', type=float)
    parser.add_argument('--dropout', type=float, default=1.0)
    parser.add_argument('--log_dir', type=str, default="logs")

    args = parser.parse_args()

    # load mnist
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # generate mnist with outlier
    mnist_outlier = MnistOutlier(mnist, args.outlier_ratio)

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
    tf.summary.scalar('accuracy', accuracy)
    with tf.Session() as sess:
        # Merge all the summaries and write them out to ./logs/ (by default)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(args.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(args.log_dir + '/test')

        sess.run(tf.global_variables_initializer())

        for i in range(args.steps):
            if i % 100 == 0:
                summary, acc = sess.run([merged, accuracy],
                                          feed_dict={x: mnist_outlier.test_images,
                                                     y: mnist_outlier.test_labels,
                                                     keep_prob: 1.0})
                test_writer.add_summary(summary, i)
                print('Accuracy at step %s: %s' % (i, acc))
            else:
                batch = mnist_outlier.next_batch(50)
                summary, _ = sess.run([merged, train_step],
                                      feed_dict={x: batch[0],
                                                 y: batch[1],
                                                 keep_prob: args.dropout})
                train_writer.add_summary(summary, i)

        train_writer.close()
        test_writer.close()

        test_accuracy = accuracy.eval(feed_dict={x: mnist_outlier.test_images,
                                                 y: mnist_outlier.test_labels,
                                                 keep_prob: 1.0})

        print('Final test accuracy {}'.format(test_accuracy))

if __name__ == '__main__':
    main()