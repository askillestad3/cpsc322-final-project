
"""
myrandomforestclassifier.py

Implements a Random Forest classifier built on top of a custom decision tree implementation. This
module provides:

1. `RandomForestDecisionTree`: A subclass of `MyDecisionTreeClassifier` from `myclassifiers.py`
   that overrides the TDIDT recursive induction procedure to support random feature selection
   (according to the `F` attribute)

2. `MyRandomForestClassifier`: An ensemble classifier that trains `N` bootstrap decision trees,
   optionally selects the top `M` trees by out-of-bag accuracy, and performs majority-vote
   prediction.

This module is based on pseudocode provided in the Final Project assignment description, authored
Dr. Sophina Luitel. It does not depend on scikit-learn.
"""

import math

import numpy as np

from .myclassifiers import MyDecisionTreeClassifier
from .myevaluation import bootstrap_sample, accuracy_score
from .myutils import get_value_frequencies, calculate_joint_entropy

class RandomForestDecisionTree(MyDecisionTreeClassifier):
    """A decision tree used within a random forest.

    This class inherits from MyDecisionTreeClassifier and overrides the `_tdidt_recursive` method
    so that each recursive split considers only `F` randomly selected attributes, following the
    provided Random Forest algorithm.

    Attributes:
        F (optional int): The number of random attributes to consider at each node split. If `F` is
            none or is greater than the number of available attributes, all attributes are used.
    """

    def __init__(self, F: int | None=None):
        """Initialize the RandomForestDecisionTree.
        
        Args:
            F (optional int): Number of random attributes to consider per split. If None or greater
                than number of available attributes, uses all attributes
        """
        super().__init__()
        self.F = F

    # "private" methods intended for internal use
    def _tdidt_recursive(
            self,
            instances: list[int],
            attributes: list[int],
            attribute_vals: list[list],
            prior_split_size: int
            ) -> list:
        """Recursively induce a decision tree using random feature selection.

        This override of the base class method performs TDIDT (top-down induction of decision
        trees), except that each recursive call restricts candidate splitting attributes to a
        random subset of the available attributes of size `F`.
        
        Args:
            instances (list of int): The list of instances in the current split of tdidt
            attributes (list of int): The indices of the attributes still available for splitting
            attribute_vals (list of list of obj): A 2D list indicating all possible values of all
                attributes where the i'th row indicates the values for the i'th attribute
            y_vals (list of obj): The list of all unique y values in response vector y_train
                (member variable)
            y_val_dict (dict from obj to int): Dictionary representing the overall distribution of
                y values (number of occurrences for each)
            prior_split_size (int): The size of the partition used in the split to make this
                recursive call, used to give the info for leaf nodes

        Returns:
            subtree (list): The subtree corresponding to the current split in TDIDT, represented in
                the project's list-based tree format.

        Raises:
            ValueError: If X_train or y_train have not been initialized.
        """
        # Verify that X_train and y_train have been initialized
        if self.X_train is None or self.y_train is None:
            raise ValueError("Cannot induce a decision tree without X_train and y_train attributes")
        
        # Find the frequency of each y value
        y_value_freq = get_value_frequencies([self.y_train[i] for i in instances])

        # Chech if there are any remaining instances
        if not instances:
            # Base case, return the leaf with the most common y globally in training data
            most_common_y = max(self._y_val_dist.items(), key=lambda t: t[1])[0]
            return ["Leaf", most_common_y, 0, prior_split_size]
        
        # Check if there are any remaining attributes
        if not attributes:
            # Find the greatest y value frequency
            greatest_y_freq = max(y_value_freq.values())

            # Find all y-values tying for the maximum frequency
            most_common_y_vals = []
            for y_val, y_freq in y_value_freq.items():
                if y_freq == greatest_y_freq:
                    most_common_y_vals.append(y_val)

            # If there is only one most common y, return the leaf node
            if len(most_common_y_vals) == 1:
                return ["Leaf", most_common_y_vals[0], greatest_y_freq, prior_split_size]
            
            # Otherwise, return the leaf with the most common tying y overall
            best_tying_y = max(most_common_y_vals, key=lambda y: self._y_val_dist[y])
            return ["Leaf", best_tying_y, greatest_y_freq, prior_split_size]
        
        # Check if all the remaining instances already belong to the same class (base case)
        if max(y_value_freq.values()) == sum(y_value_freq.values()):
            # All remaining instances belong to the same class (base case) return remaining class
            best_y = max(y_value_freq.items(), key=lambda t: t[1])[0]
            return ["Leaf", best_y, len(instances), prior_split_size]
        
        # Select a random F attributes from the attributes parameter
        if self.F is None or self.F >= len(attributes):
            reduced_attributes = attributes
        else:
            reduced_attributes = np.random.choice(attributes, self.F, replace=False)

        # Iterate across available attributes to find best E_new
        best_new_entropy = math.inf
        best_attribute = -1
        for attribute_idx in reduced_attributes:
            # Calculate the new entropy for this attribute
            new_entropy = calculate_joint_entropy(self.X_train, self.y_train, instances,
                                                  attribute_idx)
            
            # Compare this attribute's new entropy to best found so far
            if new_entropy < best_new_entropy:
                best_new_entropy = new_entropy
                best_attribute = attribute_idx

        # Instantiate and start subtree
        subtree: list = ["Attribute", f"att{best_attribute}"]

        # Get the unique values of the splitting attribute
        split_attribute_vals = attribute_vals[best_attribute]

        # Make a new list of remaining attributes after splitting on the best one
        remaining_attributes = attributes.copy()
        remaining_attributes.remove(best_attribute)

        # Make a recursive call for each value of splitting attribute
        for split_val in split_attribute_vals:
            # Find all indices with this value of splitting attribute
            split_indices = []
            for i in instances:
                if self.X_train[i][best_attribute] == split_val:
                    split_indices.append(i)

            # Add subtree for this split value to subtree (recursive call)
            subtree.append(["Value", split_val, self._tdidt_recursive(split_indices,
                                                                      remaining_attributes,
                                                                      attribute_vals,
                                                                      len(instances))])
        
        # Return the resulting subtree
        return subtree


class MyRandomForestClassifier():
    """Random Forest classifier using custom decision trees.
    
    Trains `N` bootstrap decision trees, optionally selects the top `M` by out-of-bag accuracy, and
    predicts by majority vote across the final set of trees.

    Attributes:
        N (int): Number of bootstrap decision trees to train
        M (int): Number of tres to keep. If < `N`, selects the best `M` by out-of-bag
            accuracy. If None or >= `N`, keeps all `N` trees
        F (optional int): Number of random attributes considered per split in training each tree
        X_train (list of list of obj | None): Training feature matrix
        y_train (list of obj | None): Training label vector
        trees(list of `RandomForestDecisionTree`): Final set of trained trees
        _y_val_dist (dict from obj to int): Global distribution of y-values, used to break
            prediction ties
    """

    def __init__(self, N: int, M: int | None=None, F: int | None=None):
        """Initialize the Random Forest classifier.
        
        Args:
            N (int): Number of trees to train
            M (optional int): Number of trees to keep after training. If None or >= `N`, keeps all
                trees. If < `N`, selects the top `M` using out-of-bag accuracy
            F (optional int): Number of random attributes to consider per split while training
                decision trees
        """
        self.N = N
        self.M = M if M is not None and M < N else N
        self.F = F
        self.X_train = None
        self.y_train = None
        self.trees: list[RandomForestDecisionTree] = []

        # "private" attribute
        self._y_val_dist = {}

    def fit(self, X_train: list[list], y_train: list):
        """Fit the Random Forest to the training data.
        
        For each of `N` bootstrap samples:
        - Train a `RandomForestDecisionTree`.
        - If `M` < `N`, compute its out-of-bag accuracy.
        - After all `N` trees are trained, optionally keep only the best `M`.

        Args:
            X_train (list of list of obj): Training feature matrix
            y_train (list of obj): Training label vector

        Raises:
            RuntimeError: If `M` < `N` but accuracies were not computed (not expected to occur)
        """
        # Set X_train and y_train attributes
        self.X_train = X_train
        self.y_train = y_train

        # Find distribution of y values
        for val in self.y_train:
            if val not in self._y_val_dist:
                self._y_val_dist[val] = 1
            else:
                self._y_val_dist[val] += 1

        # Generate N boostrap samples
        bootstrap_samples = [tuple(bootstrap_sample(self.X_train, self.y_train)) for _ in range(self.N)]

        # If M is less than M, make a new list for all trees and another for the accuracies
        if self.M < self.N:
            all_trees = []
            tree_accuracies = []
        else:
            all_trees = self.trees
            tree_accuracies = None

        # Iterate through all bootstrap samples, training and testing trees (if needed)
        for X_sample, X_out_of_bag, y_sample, y_out_of_bag in bootstrap_samples:
            # Train a new decision tree
            new_tree = RandomForestDecisionTree(self.F)
            new_tree.fit(X_sample, y_sample)

            # Add the new tree to the all trees list
            all_trees.append(new_tree)

            # If we're keeping track of accuracies (to narrow down to M), calculate now
            if tree_accuracies is not None:
                # Make predictions on X_out_of_bag
                y_pred = new_tree.predict(X_out_of_bag)

                # Calculate the accuracy and add to the list
                tree_accuracies.append(accuracy_score(y_out_of_bag, y_pred))

        # If we don't need to narrow down the trees, we're done here
        if self.M == self.N:
            return
        
        # Use the tree accuracies to find the indices of the best trees
        if tree_accuracies is None:
            raise RuntimeError("M is less than N but tree accuracies were not calculated")
        
        tree_accuracy_idx = list(range(self.N))
        tree_accuracy_idx.sort(key=lambda i: tree_accuracies[i], reverse=True)
        tree_accuracy_idx = tree_accuracy_idx[:self.M]
        tree_accuracy_idx.sort()

        # Add all the best trees to the trees attribute
        self.trees = [all_trees[i] for i in tree_accuracy_idx]

    def predict(self, X_test: list[list]) -> list:
        """Predict labels for test data using majority vote among decision trees.
        
        Each tree predicts independently. The final label per instance is:
        - The majority vote among all trees' predictions, or
        - In case of ties, the globally most common training label among the tying labels

        Args:
            X_test (list of list of obj): Test feature rows

        Returns:
            y_pred (list): Predicted class labels for each test instance

        Raises:
            ValueError: If predict() is called before fit()
        """
        # Verify that trees have been fitted
        if len(self.trees) == 0:
            raise ValueError("Cannot predict without fitted trees, try running fit() method first")

        # Get every tree's predictions for every test row
        tree_predictions = [tree.predict(X_test) for tree in self.trees]

        # Zip the predictions to get lists of predictions for each row
        predictions_by_row = list(zip(*tree_predictions))

        # Initialize the output prediction list
        y_pred = []

        # Get a majority vote for each test row
        for predictions in predictions_by_row:
            # Count the occurrences of each y label
            y_label_counts = {}
            for prediction in predictions:
                if prediction not in y_label_counts:
                    y_label_counts[prediction] = 1
                else:
                    y_label_counts[prediction] += 1

            # Find the maximum number of votes for any one predictions
            most_y_votes = max(y_label_counts.values())

            # Find all y values tying for the maximum number of votes
            tying_y_labels = []
            for y_label, vote_count in y_label_counts.items():
                if vote_count == most_y_votes:
                    tying_y_labels.append(y_label)

            # If there is only one tying y, predict it and continue to next row
            if len(tying_y_labels) == 1:
                y_pred.append(tying_y_labels[0])
                continue

            # Otherwise, predict the most common of the tying y labels
            y_pred.append(max(tying_y_labels, key=lambda y: self._y_val_dist[y]))

        return y_pred

