import os
import hickle
import urllib2

DATA_SET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
MUSHROOM_CLASSES = ['e', 'p']
POSSIBLE_VALUES_OF_ATTRIBUTES = [
    ['b', 'c', 'x', 'f', 'k', 's'],
    ['f', 'g', 'y', 's'],
    ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'],
    ['t', 'f'],
    ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'],
    ['a', 'd', 'f', 'n'],
    ['c', 'w', 'd'],
    ['b', 'n'],
    ['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'],
    ['e', 't'],
    ['b', 'c', 'u', 'e', 'z', 'r', '?'],
    ['f', 'y', 'k', 's'],
    ['f', 'y', 'k', 's'],
    ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],
    ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],
    ['p', 'u'],
    ['n', 'o', 'w', 'y'],
    ['n', 'o', 't'],
    ['c', 'e', 'f', 'l', 'n', 'p', 's', 'z'],
    ['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'],
    ['a', 'c', 'n', 's', 'v', 'y'],
    ['g\n', 'l\n', 'm\n', 'p\n', 'u\n', 'w\n', 'd\n']
]


def get_data_set(url):
    """Gets data set from server, if required and writes to disk.

    Args:
        url(string): A URL to the UCI Irvine mushroom classification data set

    Raises:
        ValueError: In the event that the URL is faulty or server behaves unexpectedly.

    """
    if not os.path.isfile('mushrooms.data'):
        print ("Downloading Data Set from : " + url)
        server_response = urllib2.urlopen(url)
        if server_response.getcode() != 200:
            raise ValueError("Expected 200 response from web server. Got {}".format(str(server_response.getcode())))
        with open('mushrooms.data', 'w') as output_file:
            output_file.write(server_response.read())
            print("Download successful!")


def _get_features_and_labels():
    labels = []
    feature_list = []
    read_file = open('mushrooms.data', 'r')
    print ("Reading data from disk to memory...")
    for line in read_file:
        data = line.split(',')
        label = data[0]
        features = data[1:]
        labels.append(label)
        feature_list.append(features)
    read_file.close()

    return _make_binary(feature_list, labels)


def _make_binary(feature_vectors, labels):
    binary_labels = [_get_binary_rep(label, MUSHROOM_CLASSES) for label in labels]
    binary_features = []
    print ("Creating binary representations...")
    for feature_vector in feature_vectors:
        # Iterate through all feature vectors
        binary_feature_vector = []
        for feature_no in range(22):
            feature_value = feature_vector[feature_no]
            binary_feature_vector = binary_feature_vector + _get_binary_rep(feature_value,
                                                                            POSSIBLE_VALUES_OF_ATTRIBUTES[feature_no])
        binary_features.append(binary_feature_vector)

    return binary_features, binary_labels


def _get_binary_rep(value, possible_values):
    matching = [v for v in possible_values if v == value]
    if len(matching) == 0:
        raise ValueError("Unknown Value : {}\nExpected one out of : {}".format(value, str(possible_values)))
    return [1 if val == value else 0 for val in possible_values]


def main():
    get_data_set(DATA_SET_URL)
    features, labels = _get_features_and_labels()
    features_file = open('features.hkl', mode='w')
    labels_file = open('labels.hkl', mode='w')
    print ("Writing binary representations to disk...")
    hickle.dump(features, features_file)
    hickle.dump(labels, labels_file)
    features_file.close()
    labels_file.close()

if __name__ == "__main__":
    main()
