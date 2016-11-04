import tflearn
import hickle
import os


class DataSets(object):

    """An abstraction to hold the different data sets in one place

    Attributes:
        train_features: The features in the training data set
        train_labels([float, float]): 1-hot encoded labels corresponding to each feature in the training data set
        valid_features: The features in the validation data set
        valid_labels(float, float]): 1-hot encoded labels corresponding to each feature in the validation data set
        test_features: The features in the test data set
        valid_labels(float, float]): 1-hot encoded labels corresponding to each feature in the test data set

    """

    def __init__(self, file_path):
        """Reads the files from disk."""
        self.train_features = hickle.load(open(file_path + 'train_features.hkl', 'r'))
        self.train_labels = hickle.load(open(file_path + 'train_labels.hkl', 'r'))
        self.valid_features = hickle.load(open(file_path + 'valid_features.hkl', 'r'))
        self.valid_labels = hickle.load(open(file_path + 'valid_labels.hkl', 'r'))
        self.test_features = hickle.load(open(file_path + 'test_features.hkl', 'r'))
        self.test_labels = hickle.load(open(file_path + 'test_labels.hkl', 'r'))


def build_and_test_model(datasets):
    """Builds, trains and evaluates a neural network on the given data set

    Args:
        datasets: An instance of DataSets containing the dataset to be used

    """
    # Build model
    input_layer = tflearn.input_data(shape=[None, 126])
    dense_1 = tflearn.fully_connected(input_layer, 64, activation='relu', regularizer='L2', weight_decay=0.0001)
    softmax = tflearn.fully_connected(dense_1, 2, activation='softmax')

    # Regression
    sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
    top_k = tflearn.metrics.Top_k(5)
    net = tflearn.regression(softmax, optimizer=sgd, metric=top_k, loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(datasets.train_features, datasets.train_labels, n_epoch=30, show_metric=True, run_id="dense_model",
              validation_set=(datasets.valid_features, datasets.valid_labels))

    # Test model performance
    match_count = 0.0
    predictions = model.predict(datasets.test_features)
    print len(predictions)
    for index in xrange(0, len(predictions)):
        if _compare(predictions[index], datasets.test_labels[index]):
            match_count += 1

    print ("Accuracy is : {} %".format(str(match_count / len(predictions) * 100)))


def _compare(prediction_vector, label_vector):
    """Compares two 1-hot encoded vectors.

    Returns:
        True: If both vectors resolve to the same 1-hot decoded value.
        False: In all other cases.

    """
    if (label_vector[0] > label_vector[1] and prediction_vector[0] > prediction_vector[1]) or \
            (label_vector[1] > label_vector[0] and prediction_vector[1] > prediction_vector[0]):
        return True
    else:
        return False


def main(file_path):
    build_and_test_model(DataSets(file_path))


if __name__ == "__main__":
    main(os.getcwd() + "/")
