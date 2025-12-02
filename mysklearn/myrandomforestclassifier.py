
import math

import numpy as np

from .myclassifiers import MyDecisionTreeClassifier
from .myevaluation import bootstrap_sample, accuracy_score
from .myutils import calculate_entropy

class RandomForestDecisionTree(MyDecisionTreeClassifier):
    def __init__(self, F: int | None=None):
        super().__init__()
        self.F = F

    # "private" methods intended for internal use
    def _tdidt_recursive(
            self,
            instances: list[int],
            attributes: list[int],
            attribute_vals: list[list],
            y_vals: list,
            prior_split_size: int
            ) -> list:
        """Recursive function to perform top-down induction of a decision tree. Identical to that
        used by MyDecisionTreeClassfier except that each iteration randomly selects F random
        candidate attributes
        
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
            subtree (list): The subtree corresponding to the current split in TDIDT
        """
        # Verify that X_train and y_train have been initialized
        if self.X_train is None or self.y_train is None:
            raise ValueError("Cannot induce a decision tree without X_train and y_train attributes")
        
        # Check if there are any remaining attributes
        if not attributes:
            # Base case, find the frequency of each y value
            y_value_freq = {val: 0 for val in y_vals}
            for i in instances:
                y_value_freq[self.y_train[i]] += 1

            # Find the greatest frequency
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
        y_value_freq = {val: 0 for val in y_vals}
        for i in instances:
            y_value_freq[self.y_train[i]] += 1

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
            new_entropy = calculate_entropy(self.X_train, self.y_train, instances,
                                            attribute_idx, attribute_vals[attribute_idx],
                                            y_vals)
            
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
                                                                      attribute_vals, y_vals,
                                                                      len(instances))])
        
        # Return the resulting subtree
        return subtree


class MyRandomForestClassifier():
    def __init__(self, N: int, M: int | None=None, F: int | None=None):
        self.N = N
        self.M = M if M is not None and M < N else N
        self.F = F
        self.X_train = None
        self.y_train = None
        self.trees: list[RandomForestDecisionTree] = []

    def fit(self, X_train: list[list], y_train: list):
        # Set X_train and y_train attributes
        self.X_train = X_train
        self.y_train = y_train

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

    def predict(self, X_test) -> list:
        return []

