import os
import hickle
from multiprocessing import Pool

# Set division ratio of the data set.
TRAIN = 9
VALIDATION = 1
TEST = 1


def _load_data_set_in_memory(feature_file_path, labels_file_path):
    """Reads data set from disk to memory."""
    if not os.path.isfile(feature_file_path):
        raise ValueError("Invalid Feature file Path")
    if not os.path.isfile(labels_file_path):
        raise ValueError("Invalid labels file Path")
    print ("Reading features from disk to memory")
    features = hickle.load(open(feature_file_path, 'r'))
    print ("Reading labels from disk to memory")
    labels = hickle.load(open(labels_file_path, 'r'))

    return features, labels


def _split_data_sets(features, labels):
    """Splits data set in the ratio set by the global constants
    
    Raises:
        ValueError: If the data set contains an invalid label
    """
    train_features = []
    valid_features = []
    test_features = []
    train_labels = []
    valid_labels = []
    test_labels = []
    
    train_count_0 = 0    
    valid_count_0 = 0
    test_count_0 = 0
    train_count_1 = 0
    valid_count_1 = 0
    test_count_1 = 0

    total_entries = len(labels)
    print ("Splitting the data set")
    for index in range(total_entries):
        label = labels[index]
        if label == [0, 1]:
            if train_count_0 < TRAIN:
                train_features.append(features[index])
                train_labels.append(label)
                train_count_0 += 1
            elif test_count_0 < TEST:
                test_features.append(features[index])
                test_labels.append(label)
                test_count_0 += 1
            elif valid_count_0 < VALIDATION:
                valid_features.append(features[index])
                valid_labels.append(label)
                valid_count_0 += 1
            else:
                # Reset all counters
                test_count_0 = 0
                train_count_0 = 0
                valid_count_0 = 0
        elif label == [1, 0]:
            if train_count_1 < TRAIN:
                train_features.append(features[index])
                train_labels.append(label)
                train_count_1 += 1
            elif test_count_1 < TEST:
                test_features.append(features[index])
                test_labels.append(label)
                test_count_1 += 1
            elif valid_count_1 < VALIDATION:
                valid_features.append(features[index])
                valid_labels.append(label)
                valid_count_1 += 1
            else:
                # Reset all counters 
                test_count_1 = 0
                train_count_1 = 0
                valid_count_1 = 0
        else:
            raise ValueError("Invalid Data label : {}".format(label))

    # Print sizes on screen
    print ("Data set sizes:\nTrain: {}\nTest: {}\nValidation: {}".format(str(len(train_features)),
                                                                         str(len(test_features)),
                                                                         str(len(valid_features))))
    # Write data to file
    data_file_mappings = (('train_features.hkl', train_features),
                          ('train_labels.hkl', train_labels),
                          ('valid_features.hkl', valid_features),
                          ('valid_labels.hkl', valid_labels),
                          ('test_features.hkl', test_features),
                          ('test_labels.hkl', test_labels)
                          )
    write_pool = Pool(6)
    write_pool.map(_writer, data_file_mappings)


def _writer(mapping):
    """Writes data to file specified by file path.
    
    Args:
        mapping(string, Iterable) : Tuple with path of the file to write to (1st element) and the data (2nd element)
                
    """
    file_path = mapping[0]
    data = mapping[1]
    print ("Start writing to : {}".format(file_path))
    hickle.dump(data, open(file_path, 'w'))
    print ("Done writing to : {}".format(file_path))    


def main():
    all_features, all_labels = _load_data_set_in_memory('features.hkl', 'labels.hkl')
    _split_data_sets(all_features, all_labels)


if __name__ == "__main__":
    main()
