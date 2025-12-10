
import math
from tabulate import tabulate

import numpy as np # use numpy's random number generation

from .myutils import calculate_entropy, calculate_joint_entropy

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    # Verify X and y have same length
    if len(X) != len(y):
        raise ValueError('X and y must have the same length')

    # If a random state is provided, seed the random number generator
    if random_state is not None:
        np.random.seed(random_state)
    
    # If shuffle is true, perform a parallel shuffle on X and y
    if shuffle is True:
        # Randomize the indices
        indices = list(range(len(X)))
        np.random.shuffle(indices)

        # Define new lists
        X_new, y_new = [], []

        # Use randomized indices to shuffle lists
        for i in indices:
            X_new.append(X[i])
            y_new.append(y[i])

        # Ensure that shuffle sufficiently disrupted order (only for unit test)
        for i in range(len(X)):
            # Check if value found in same place
            if X_new[i] == X[i]:
                # Find first different value (to avoid swapping an equivalent input)
                j = i + 1
                while j < len(X) and X_new[j] == X_new[i]:
                    j += 1
                
                # Perform the swap
                X_new[i], X_new[j] = X_new[j], X_new[i]
                y_new[i], y_new[j] = y_new[j], y_new[i]
        
        X, y = X_new, y_new
    
    # If test size is a float, calculate the int number of test instances
    if isinstance(test_size, float):
        num_test_instances = math.ceil(len(X) * test_size)
    else:
        num_test_instances = test_size

    # If the number of test instances is less than 0, make it 0
    if num_test_instances < 0:
        num_test_instances = 0

    # If the number of test instances is too great for the data, drop to number of rows
    if num_test_instances > len(X):
        num_test_instances = len(X)

    # Calculate the number of training instances
    num_train_instances = len(X) - num_test_instances

    # Use slice indexing to make the partition
    X_train = X[:num_train_instances]
    y_train = y[:num_train_instances]
    X_test = X[num_train_instances:]
    y_test = y[num_train_instances:]

    return X_train, X_test, y_train, y_test
    

def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    # If a random state is provided, seed the random number generator
    if random_state is not None:
        np.random.seed(random_state)

    # Generate a list of X's indices
    X_indices = list(range(len(X)))
    
    # If shuffle is true, let numpy shuffle X
    if shuffle is True:
        # Shuffle the list of indices
        np.random.shuffle(X_indices)

        # Pass through the indices to ensure none is in the same place
        for i, index in enumerate(X_indices):
            if index == i:
                if i == len(X_indices) - 1:
                    X_indices[0], X_indices[i] = X_indices[i], X_indices[0]
                else:
                    X_indices[i], X_indices[i + 1] = X_indices[i + 1], X_indices[i]

    # Calculate the number of "big folds" needed
    big_folds = len(X_indices) % n_splits

    # Calculate small fold size (big folds one element bigger)
    small_fold_size = len(X_indices) // n_splits

    # Find the split points (as first index of fold)
    split_points = []
    i = 0
    while len(split_points) < n_splits:
        split_points.append(i)
        if big_folds > 0:
            i += small_fold_size + 1
            big_folds -= 1
        else:
            i += small_fold_size

    # List to store train and test indices for folds
    folds = []

    # Create a fold for each split point
    for i, split_start in enumerate(split_points):
        # Find the ending of this split
        if i == len(split_points) - 1:
            split_end = len(X)
        else:
            split_end = split_points[i + 1]

        # Set the indices of this split as the testing set
        test_indices = X_indices[split_start:split_end]

        # Set all other valid indices as the training set
        train_indices = X_indices[:split_start] + X_indices[split_end:]

        # Add the train and test indices to the folds list
        folds.append((train_indices, test_indices))

    return folds


# BONUS function
def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    # If a random state was provided, seed the random number generator
    if random_state is not None:
        np.random.seed(random_state)

    # For each response variable in y, find all occurrence indices
    y_occurrence_indices = {}
    for i, val in enumerate(y):
        if val in y_occurrence_indices:
            y_occurrence_indices[val].append(i)
        else:
            y_occurrence_indices[val] = [i]

    # If shuffle is true, shuffle the index list for each y value
    if shuffle is True:
        for index_list in y_occurrence_indices.values():
            np.random.shuffle(index_list)

    # Instantiate return data structure
    folds = [([], []) for _ in range(n_splits)]

    # Cycle through folds, giving indexes from the y occurrence indices to test set
    # Start on first fold
    fold_index = 0
    for index_list in y_occurrence_indices.values():
        for index in index_list:
            # Assign index to test set of fold
            folds[fold_index][1].append(index)

            # Move to next fold
            fold_index = (fold_index + 1) % n_splits

    # For each fold, give all indices not in training set to test set
    for fold in folds:
        # Get the list (in order) of all indices not in test set
        for index in range(len(X)):
            if index not in fold[1]:
                fold[0].append(index)

        # If shuffle is true, shuffle the list
        if shuffle is True:
            np.random.shuffle(fold[0])

    return folds


def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    # If y is not none, ensure it has the same length as X
    if y is not None and len(X) != len(y):
        raise ValueError("Parameters X and y must have the same length")
    
    # If a random state is provided, seed the random number generator
    if random_state is not None:
        np.random.seed(random_state)

    # If n_samples is none, set to the first dimension of X
    if n_samples is None:
        n_samples = len(X)

    # Initialize the return lists
    X_sample, X_out_of_bag, y_sample, y_out_of_bag = [], [], [], []

    # Select a number of indices equal to the length of X with replacement
    sample_indices = np.random.choice(len(X), size=n_samples, replace=True)

    # Find the out of bag indices
    out_of_bag_index_set = set(range(len(X)))
    for i in sample_indices:
        out_of_bag_index_set.discard(i)

    out_of_bag_indices = list(out_of_bag_index_set)

    # Use the indices to generate X_samples and X_out_of_bag
    for i in sample_indices:
        X_sample.append(X[i])

    for i in out_of_bag_indices:
        X_out_of_bag.append(X[i])

    # If y is not none, use the indices to generate y_samples and y_out_of_bag
    if y is not None:
        for i in sample_indices:
            y_sample.append(y[i])

        for i in out_of_bag_indices:
            y_out_of_bag.append(y[i])

    # If y is none, make both of its sample lists none
    if y is None:
        y_sample, y_out_of_bag = None, None

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag


def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    # Ensure y_true and y_pred have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("Parameters y_true and y_pred must have the same length")

    # Assign each label an index for the confusion matrix
    label_index = {label: i for i, label in enumerate(labels)}

    # Initialize the confusion matrix with 0 for each entry
    confusion_matrix = [[0] * len(labels) for _ in range(len(labels))]

    # For each true, pred pair, increment the corresponding value in the confusion matrix
    for true, pred in zip(y_true, y_pred):
        confusion_matrix[label_index[true]][label_index[pred]] += 1

    return confusion_matrix


def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    # Ensure y_true and y_pred are the same length
    if len(y_true) != len(y_pred):
        raise ValueError("Parameters y_true and y_pred must have the same length")

    # Find the total number of correct predictions
    num_correct_preds = sum([1 if true == pred else 0 for true, pred in zip(y_true, y_pred)])

    # If told to normalize, divide by y_true's length, otherwise return number of correct preds
    if normalize is True:
        return num_correct_preds / len(y_true)
    else:
        return num_correct_preds

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    # Verify that y_true and y_pred have the same nonzero length
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("Parameters y_true and y_pred must have the same, nonzero length")
    
    # Verify that labels, if given, isn't an empty list
    if labels is not None and len(labels) == 0:
        raise ValueError("Parameter labels cannot be an empty list")
    
    # If no labels given, find them based on unique values in y_true
    if labels is None:
        labels = list(set(y_true))

    # If no positive label given, set as the first label in labels
    if pos_label is None:
        pos_label = labels[0]

    # Count the true and false positives
    tp, fp = 0, 0
    for true, pred in zip(y_true, y_pred):
        if pred == pos_label:
            if true == pos_label:
                tp += 1
            else:
                fp += 1

    # Give a precision of 0 (default value) if positive never predicted
    if tp + fp == 0:
        return 0.0

    # Calculate and return the precision score
    return tp / (tp + fp)


def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    # Verify that y_true and y_pred have the same nonzero length
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("Parameters y_true and y_pred must have the same, nonzero length")
    
    # Verify that labels, if given, isn't an empty list
    if labels is not None and len(labels) == 0:
        raise ValueError("Parameter labels cannot be an empty list")
    
    # If no labels given, find them based on unique values in y_true
    if labels is None:
        labels = list(set(y_true))

    # If no positive label given, set as the first label in labels
    if pos_label is None:
        pos_label = labels[0]

    # Count the true and false positives
    tp, fn = 0, 0
    for true, pred in zip(y_true, y_pred):
        if true == pos_label:
            if pred == pos_label:
                tp += 1
            else:
                fn += 1

    # Give a precision of 0 (default value) if no positives in y_true
    if tp + fn == 0:
        return 0.0

    # Calculate and return the recall score
    return tp / (tp + fn)


def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    # NOTE: This is not the most efficient method, but it's minimally less efficient and far easier

    # Verify that y_true and y_pred have the same nonzero length
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("Parameters y_true and y_pred must have the same, nonzero length")
    
    # Verify that labels, if given, isn't an empty list
    if labels is not None and len(labels) == 0:
        raise ValueError("Parameter labels cannot be an empty list")
    
    # If no labels given, find them based on unique values in y_true
    if labels is None:
        labels = list(set(y_true))

    # If no positive label given, set as the first label in labels
    if pos_label is None:
        pos_label = labels[0]

    # Compute the precision and recall using existing functions
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)

    # Return 0 (default value) if precision and recall are both zero
    if precision + recall == 0:
        return 0.0

    # Compute and return the F1 score
    return 2 * (precision * recall) / (precision + recall)


def classification_report(y_true: list, y_pred: list, labels=None, output_dict: bool=False) -> str | dict:
    """Build a text report or a dictionary showing the main classification metrics.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        output_dict(bool): If True, return output as dict instead of a str

    Returns:
        report(str or dict): Summary of the precision, recall, F1 score for each class.
            Dictionary returned if output_dict is True. Dictionary has the following structure:
                {'label 1': {'precision':0.5,
                            'recall':1.0,
                            'f1-score':0.67,
                            'support':1},
                'label 2': { ... },
                ...
                }
            The reported averages include macro average (averaging the unweighted mean per label) and
            weighted average (averaging the support-weighted mean per label).
            Micro average (averaging the total true positives, false negatives and false positives)
            multi-class with a subset of classes, because it corresponds to accuracy otherwise
            and would be the same for all metrics. 

    Notes:
        Loosely based on sklearn's classification_report():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    """
    # NOTE: We're going to form the dict for all runs, then convert to the multiline string only if output_dict is false
    
    # Verify that y_true and y_pred have the same nonzero length
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("Parameters y_true and y_pred must have the same, nonzero length")
    
    # Verify that labels, if given, isn't an empty list
    if labels is not None and len(labels) == 0:
        raise ValueError("Parameter labels cannot be an empty list")
    
    # If no labels given, find them based on unique values in y_true
    if labels is None:
        labels = list(set(y_true))

    # Instantiate report dictionary
    report_dict = {}

    # Compute the accuracy
    report_dict["accuracy"] = accuracy_score(y_true, y_pred)

    # Treating each label as a positive label, compute the precision, recall, f1, and support
    for label in labels:
        # Instantiate new sub-dict
        report_dict[label] = {}

        # Compute the metrics
        report_dict[label]["precision"] = binary_precision_score(y_true, y_pred, labels=labels, pos_label=label)
        report_dict[label]["recall"] = binary_recall_score(y_true, y_pred, labels=labels, pos_label=label)
        report_dict[label]["f1-score"] = binary_f1_score(y_true, y_pred, labels=labels, pos_label=label)

        # Find the support (number of occurrences of label in y_true)
        report_dict[label]["support"] = sum(1 if y == label else 0 for y in y_true)

    # Compute the macro avg for all evaluation metrics
    report_dict["macro avg"] = {}
    report_dict["macro avg"]["precision"] = sum(report_dict[label]["precision"] for label in labels) / len(labels)
    report_dict["macro avg"]["recall"] = sum(report_dict[label]["recall"] for label in labels) / len(labels)
    report_dict["macro avg"]["f1-score"] = sum(report_dict[label]["f1-score"] for label in labels) / len(labels)

    # Compute the weighted avg (by support) for all evaluation metrics
    report_dict["weighted avg"] = {}
    report_dict["weighted avg"]["precision"] = sum(report_dict[label]["support"] * report_dict[label]["precision"] for label in labels) \
        / sum(report_dict[label]["support"] for label in labels)
    report_dict["weighted avg"]["recall"] = sum(report_dict[label]["support"] * report_dict[label]["recall"] for label in labels) \
        / sum(report_dict[label]["support"] for label in labels)
    report_dict["weighted avg"]["f1-score"] = sum(report_dict[label]["support"] * report_dict[label]["f1-score"] for label in labels) \
        / sum(report_dict[label]["support"] for label in labels)
    
    # Add the total support to the macro and weighted avg
    report_dict["macro avg"]["support"] = report_dict["weighted avg"]["support"] = sum(report_dict[label]["support"] for label in labels)
    
    # If output is a dictionary, return the report dict now
    if output_dict is True:
        return report_dict
    
    # Instantiate a 2d list table to represent the report (will be converted to tabulated string)
    report_table = [["","precision","recall","f1-score","support"], [""]]

    # Add each label's metrics to the report table
    for label in labels:
        report_table.append([label, f"{report_dict[label]['precision']:.2f}", f"{report_dict[label]['recall']:.2f}",
                             f"{report_dict[label]['f1-score']:.2f}", report_dict[label]['support']])
        
    # Insert a blank line in the table
    report_table.append([])
        
    # Add a partial row for the accuracy (taking support from "macro avg")
    report_table.append(["accuracy", "", "", f"{report_dict['accuracy']:.2f}", str(report_dict["macro avg"]["support"])])

    # Add the macro avg and weighted avg rows to the report table
    report_table.append(["macro avg", f"{report_dict['macro avg']['precision']:.2f}", f"{report_dict['macro avg']['recall']:.2f}",
                         f"{report_dict['macro avg']['f1-score']:.2f}", str(report_dict["macro avg"]["support"])])
    report_table.append(["weighted avg", f"{report_dict['weighted avg']['precision']:.2f}", f"{report_dict['weighted avg']['recall']:.2f}",
                         f"{report_dict['weighted avg']['f1-score']:.2f}", str(report_dict["weighted avg"]["support"])])
    
    # Tabulate and return the report table
    return tabulate(report_table, tablefmt="plain")


def mutual_information(X: list, Y: list) -> float:
    """Calculates the mutual information between categorical vectors X and Y.
    
    Args:
        X (list of obj): The first list of categorical values, typically the predictor
        Y (list of obj): The second list of categorical values, typically the response
    
    Return:
        I (float): The mutual information between X and Y, equivalent to the information gain
            between entropy and joint entropy

    Notes:
        This function is modeled off the equation in terms of entropy on this Wikipedia page:
        https://en.wikipedia.org/wiki/Mutual_information
    """
    # Verify that X and Y are the same length
    if len(X) != len(Y):
        raise ValueError("Categorical vectors X and Y must be the same length")
    
    # Convert X into a 2D predictor matrix
    X_matrix = [[x_val] for x_val in X]

    # Calculate and return the mutual information
    return calculate_entropy(Y) - calculate_joint_entropy(X_matrix, Y, list(range(len(X))), 0)


def pearson_r(X: list, Y: list) -> float:
    """Calculates the Pearson Correlation Coefficient (r) between two numerical columns.
    
    Args:
        X (list of obj): The first list of numeric values, typically the predictor
        Y (list of obj): The second list of numeric values, typically the response

    Returns:
        r (float): The Pearson correlation coefficient, ranging from -1 to 1 with -1 representing
            a perfect negative correlation, 0 representing no correlation, and 1 representing a perfect
            positive correlation

    Notes:
        - In the demonstration, this function is used for a binary response variable. We simply
          code the binary variable as 0/1 and calculate as normal. There may be other methods more
          appropriate for correlation with a binary response
        - Pearson's Correlation Coefficient is a pretty standard statistical measure, but I grabbed
          my equations from the following Wikipedia pages:
          https://en.wikipedia.org/wiki/Pearson_correlation_coefficient AND
          https://en.wikipedia.org/wiki/Covariance
    """
    # Verify that both lists have an equal, nonzero length
    if len(X) == 0:
        raise ValueError("Cannot calculate pearson r with empty list X")
    if len(Y) == 0:
        raise ValueError("Cannot calculate pearson r with empty list Y")
    if len(X) != len(Y):
        raise ValueError("Cannot calculate pearson r between lists of different length")

    # Calculate the mean values for X and Y
    mean_X = sum(X) / len(X)
    mean_Y = sum(Y) / len(Y)

    # Calculate the standard deviations for X and Y
    std_X = float(np.std(X))
    std_Y = float(np.std(Y))

    # Calculate the list of pairwise products between X and Y
    X_Y_pairwise = [x * y for x, y in zip(X, Y)]

    # Calculate the mean pairwise product
    mean_X_Y = sum(X_Y_pairwise) / len(X_Y_pairwise)

    # Calculate covariance
    covariance = mean_X_Y - mean_X * mean_Y

    # Calculate and return pearson correlation coefficient
    return covariance / (std_X * std_Y)


def relieff_score(X: list[list], y: list, k: int) -> list[float]:
    """Performs the ReliefF algorithm to find the usefulness of each feature in X at separating the
    class labels in y.
    
    Args:
        X (list of list of obj): The predictor matrix, where each row is an instance containing the
            same number of standardized or normalized numeric features
        y (list of obj): The response vector, assumed to be categorical
        k (int): The number of nearest instances of the same and different class labels to consider
            in calculating the score
    
    Returns:
        weights (list of float): The score for each feature in X, which may be positive or negative,
            higher values indicating a better discriminator of class labels
    
    Notes:
        - It is assumed that all rows of X are the same length (algorithm will break if not)
        - It is assumed that there are at least k + 1 instances of each class label in y (algorithm
          will break if not)
    """
    # Verify that X and y are the same, nonzero length
    if len(X) == 0:
        raise ValueError("Cannot calculate reliefF score with empty list X")
    if len(y) == 0:
        raise ValueError("Cannot calculate reliefF score with empty list y")
    if len(X) != len(y):
        raise ValueError("Cannot calculate reliefF score between X and y of different length")
    
    # Find the dimensions of X, n and m
    n, m = len(X), len(X[0])

    # Set the initial weight vector
    weights = [0.0 for _ in range(m)]

    # Iterate through rows of X and labels in y
    for i in range(n):
        row, label = X[i], y[i]

        # Find the L1 norm distance from this row to all other rows
        distances = [sum(row[k] - X[j][k] for k in range(m)) for j in range(n)]

        # Find the near hits (k nearestrows with same label)
        near_hits = sorted(range(n), key=lambda j: distances[j] if j != i and y[j] == label else math.inf)[:k]

        # Find the near misses (same but different labels)
        near_misses = sorted(range(n), key=lambda j: distances[j] if j != i and y[i] != label else math.inf)[:k]

        # Update the weights according to the near hits and near misses
        for j in range(m):
            for near_hit in near_hits:
                weights[j] -= abs(row[j] - X[near_hit][j]) / k
            for near_miss in near_misses:
                weights[j] += abs(row[j] - X[near_miss][j]) / k

    return weights

