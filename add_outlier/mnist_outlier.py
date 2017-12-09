import numpy as np


class MnistOutlier:
    def __init__(self, mnist, outlier_ratio):
        self.mnist = mnist
        self.outlier_ratio = outlier_ratio
        self.train_images, self.train_labels = self.generate_outlier_mnist()
        self.test_images, self.test_labels = self.mnist.test.images, self.mnist.test.labels
        self.validation_images, self.validation_labels = self.mnist.validation.images, self.mnist.validation.labels
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._num_examples = self.train_images.shape[0]

    def generate_outlier_mnist(self):
        train_images = self.mnist.train.images
        train_labels = self.mnist.train.labels
        label_numbers = np.where(train_labels == 1)[1]
        mixed_images = np.array([])
        mixed_labels = np.array([])
        for i in range(10):
            i_positions = np.where(label_numbers == i)[0].reshape(-1)
            i_images = train_images[i_positions]
            i_quantity = i_images.shape[0]
            outlier_quantity = int(i_quantity * self.outlier_ratio)
            no_outlier_quantity = i_quantity - outlier_quantity
            list_remove_i = range(10)
            list_remove_i.remove(i)
            outlier_labels = np.random.choice(list_remove_i, outlier_quantity, replace=True)
            if mixed_images.shape[0] == 0:
                mixed_images = i_images
                mixed_labels = np.concatenate((outlier_labels, np.array([i] * no_outlier_quantity)))
            else:
                mixed_images = np.concatenate((mixed_images, i_images))
                mixed_labels = np.concatenate((mixed_labels, outlier_labels))
                mixed_labels = np.concatenate((mixed_labels, np.array([i] * no_outlier_quantity)))

        idx = np.random.permutation(len(mixed_labels))
        mixed_images, mixed_labels = mixed_images[idx], mixed_labels[idx]

        return mixed_images, self.dense_to_one_hot(mixed_labels, 10)

    def dense_to_one_hot(self, labels, num_classes):
        num_labels = labels.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels.ravel()] = 1
        return labels_one_hot

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self.train_images[start:self._num_examples]
            labels_rest_part = self.train_labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self.train_images = self.train_images[perm]
                self.train_labels = self.train_labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self.train_images[start:end]
            labels_new_part = self.train_labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self.train_images[start:end], self.train_labels[start:end]