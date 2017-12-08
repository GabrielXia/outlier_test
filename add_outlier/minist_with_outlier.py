import numpy as np


class MnistWithOutlier:

    def __init__(self, mnist, no_outlier_numbers, outlier_numbers, outlier_ratio=0.3):
        self.mnist = mnist
        self.no_outlier_numbers = no_outlier_numbers
        self.outlier_numbers = outlier_numbers
        self.outlier_ratio = outlier_ratio
        self.train_images, self.train_labels = self.add_outlier(self.mnist.train.images, self.mnist.train.labels)
        self.test_images, self.test_labels = self.extract_test(self.mnist.test.images, self.mnist.test.labels)
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._num_examples = self.train_images.shape[0]

    def extract_test(self, images, labels):
        extracted_images, extracted_labels = self.extract_number(images, labels)
        extracted_labels = self.dense_to_one_hot(extracted_labels, len(self.no_outlier_numbers))
        return extracted_images, extracted_labels

    def extract_number(self, images, labels):
        mixed_images = np.array([])
        mixed_labels = np.array([])
        label_numbers = np.where(labels == 1)[1]
        for i in self.no_outlier_numbers:
            i_positions = np.where(label_numbers == i)[0].reshape(-1)
            if mixed_images.shape[0] == 0:
                mixed_images = images[i_positions]
            else:
                mixed_images = np.concatenate((mixed_images, images[i_positions]))
            mixed_labels = np.concatenate((mixed_labels, np.array([i] * len(i_positions))))
        return mixed_images, mixed_labels

    def add_outlier(self, images, labels):
        mixed_images, mixed_labels = self.extract_number(images, labels)
        if self.outlier_ratio > 0:
            outlier_num = int(mixed_images.shape[0] * self.outlier_ratio)
            outlier_images = np.array([])
            label_numbers = np.where(labels == 1)[1]
            for j in self.outlier_numbers:
                j_positions = np.where(label_numbers == j)[0].reshape(-1)
                if outlier_images.shape[0] == 0:
                    outlier_images = images[j_positions]
                else:
                    outlier_images = np.concatenate((outlier_images, images[j_positions]))

            outlier_images_samples = outlier_images[np.random.choice(len(outlier_images), outlier_num, replace=False)]
            outlier_labels_samples = np.random.choice(self.no_outlier_numbers, outlier_num, replace=True)

            mixed_images = np.concatenate((mixed_images, outlier_images_samples))
            mixed_labels = np.concatenate((mixed_labels, outlier_labels_samples))

            idx = np.random.permutation(len(mixed_labels))
            mixed_images, mixed_labels = mixed_images[idx], mixed_labels[idx]

        return mixed_images, self.dense_to_one_hot(mixed_labels, len(self.no_outlier_numbers))

    def dense_to_one_hot(self, labels, num_classes):
        labels_dense = np.array([self.no_outlier_numbers.index(i) for i in labels])
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
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