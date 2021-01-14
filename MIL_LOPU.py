import numpy as np
from misvmio import parse_c45, bag_set
import misvm
import random


def get_dataset():
    def get_raw_data():
        # Load list of C4.5 Examples
        example_set = parse_c45('musk1')

        # Get stats to normalize data
        raw_data = np.array(example_set.to_float())
        data_mean = np.average(raw_data, axis=0)
        data_std  = np.std(raw_data, axis=0)
        data_std[np.nonzero(data_std == 0.0)] = 1.0
        def normalizer(ex):
            ex = np.array(ex)
            normed = ((ex - data_mean) / data_std)
            # The ...[:, 2:-1] removes first two columns and last column,
            # which are the bag/instance ids and class label, as part of the
            # normalization process
            return normed[2:-1]
        # Group examples into bags
        bagset = bag_set(example_set)
        # Convert bags to NumPy arrays
        bags = [np.array(b.to_float(normalizer)) for b in bagset]
        labels = [b.label for b in bagset]

        tmp = list(zip(bags, labels))
        random.seed(55)
        random.shuffle(tmp)
        bags, labels = zip(*tmp)
        return bags, labels

    bags, labels = get_raw_data()
    num_test = 10
    num_val = 10
    train_bags, train_labels = bags[num_test:-num_val], labels[num_test:-num_val]
    val_bags, val_labels = bags[-num_val:], labels[-num_val:]
    test_bags, test_labels = bags[:num_test], labels[:num_test]
    return (train_bags, train_labels), (val_bags, val_labels), (test_bags, test_labels)


def convert_bags_to_instances(bags, labels):
    num_instances_list = [bag.shape[0] for bag in bags]
    ilabels = [[label]*n for label, n in zip(labels, num_instances_list)]

    instances = np.concatenate(bags, axis=0)
    ilabels = np.concatenate(ilabels, axis=0)

    print('bags and labels shape', instances.shape, ilabels.shape)
    return (instances, ilabels)


class Classifier:
    # def __init__(self, train_bags, train_labels):
    #     train_labels = np.array(train_labels) * 2 - 1
    #     test_labels = np.array(test_labels) * 2 - 1
    #     svm = misvm.MissSVM(kernel='linear', C=1.0, max_iters=20)
    #     svm.fit(train_bags, train_labels)
    #     predictions = svm.predict(test_bags)
    #     acc2 = np.average(test_labels == np.sign(predictions))

    def __init__(self, train_bags, train_labels):
        self.train_bags = train_bags
        self.train_labels = train_labels

        self.train_instances, self.train_ilabels = convert_bags_to_instances(train_bags, train_labels)
        self.K = self.get_number_of_k_nearest_neighbors()
        
    def get_number_of_k_nearest_neighbors(self):
        # return k 
        # calculated from training data
        n_instances = self.train_instances.shape[0]
        n_bags = len(self.train_bags)
        return round(n_instances / n_bags / 2.0)

    def predict(self, x):
        # Equation 8: P(+|x)
        # x.shape = 1*166
        # intances.shape = num_instances * 166 
        # ilabels.shape: num_instance
        dis = np.sum(np.power((self.train_instances-x), 2), axis=1)
        res = sorted(list(zip(dis, self.train_ilabels)), key=lambda x: x[0])
        res = np.array(res)
        res = res[:self.K]
        num_pos = np.sum(res[:, 1] > 0)
        return num_pos / float(self.K)


def get_beta(labels):
    # Equation 2: P(+) = beta
    labels = np.array(labels)
    return np.sum(labels > 0) / labels.shape[0]


def get_theta_hat(val_instances, val_ilabels):
    # Equation 11
    theta = 0.0
    num_neg = 0
    global classifier
    for x, label in zip(val_instances, val_ilabels):
        if label == 0:
            num_neg += 1
            theta += 1 - classifier.predict(x)
    res = theta / num_neg
    return res


def get_max_posterior_pos(val_instances, val_ilabels):
    # Equatin 13: 2 line below
    global classifier
    theta = -1e9
    for x, label in zip(val_instances, val_ilabels):
        if label > 0:
            theta = max(theta, classifier.predict(x))
    return theta


def get_t(val_instances, val_ilabels):
    theta_hat = get_theta_hat(val_instances, val_ilabels)
    max_posterior = get_max_posterior_pos(val_instances, val_ilabels)
    return (1- theta_hat + max_posterior) / 2.0


def get_number_of_pos_instances_with_condition(val_instances, val_ilabels, t):
    # Equation 13: numerator
    # Notes: compute using validation set
    global classifier
    counter = 0
    for x, label in zip(val_instances, val_ilabels):
        if label > 0:
            if classifier.predict(x) > t:
                counter += 1
    return counter


def get_number_of_pos_instances(val_instances, val_ilabels):
    # Equation 13: denominator
    # Notes: compute using validation set
    return np.sum(val_ilabels > 0)


def get_alpha_hat(val_instances, val_ilabels):
    t = get_t(val_instances, val_ilabels)
    numerator = get_number_of_pos_instances_with_condition(val_instances, val_ilabels, t)
    denominator = get_number_of_pos_instances(val_instances, val_ilabels)
    res = numerator / float(denominator)
    return res


def is_concept(x, beta, alpha):
    global classifier
    k_pos = classifier.predict(x)
    k_neg = 1 - k_pos
    tmp = (1 + beta - 2*alpha*beta) / (1 - beta)
    if k_pos >= k_neg * tmp:
        return 1
    return 0


def is_bag_pos(new_bag, alpha, beta):
    concept_counter = 0
    for instance in new_bag:
        concept_counter += is_concept(instance, alpha=0.1, beta=0.1)
    fraction = concept_counter / new_bag.shape[0]
    threshold = (1-alpha*beta)*(1-2*beta)/(2*(1-beta)*new_bag.shape[0]) + alpha*beta
    return fraction > threshold


def main():
    # Khoi tao du lieu 
    train_data, val_data, test_data = get_dataset()
    
    train_bags, train_labels = train_data
    val_bags, val_labels = val_data
    test_bags, test_labels = test_data

    train_instances, train_ilabels = convert_bags_to_instances(train_bags, train_labels)
    val_instances, val_ilabels = convert_bags_to_instances(val_bags, val_labels)
    # global K
    # K = get_number_of_k_nearest_neighbors(train_instances, train_bags)
    # print('Number of K nearest neighbors:', K)
    # training
    global classifier
    classifier = Classifier(train_bags, train_labels)

    ALPHA = get_alpha_hat(val_instances, val_ilabels)
    BETA = get_beta(train_ilabels)

    # testing
    acc = 0
    for bag, blabel in zip(test_bags, test_labels):
        bpred = is_bag_pos(bag, ALPHA, BETA)
        acc += (blabel == bpred)
    print('Accuracy:', acc / len(test_bags))

    train_labels = np.array(train_labels) * 2 - 1
    test_labels = np.array(test_labels) * 2 - 1
    svm = misvm.MissSVM(kernel='linear', C=1.0, max_iters=20)
    svm.fit(train_bags, train_labels)
    predictions = svm.predict(test_bags)
    acc2 = np.average(test_labels == np.sign(predictions))
    print(acc2)


if __name__ == '__main__':
    main()