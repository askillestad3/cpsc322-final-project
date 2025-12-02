
from mysklearn import myutils
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor

"""
Programmer: Miles Mercer (skeleton code by Dr. Luitel)
Class: CPSC 322, Fall 2025
Programming Assignment #7
11/30/25
Description: This file implements a decision tree classifier object, with fitting and predicting
             methods as well as additional methods to display the decision rules and visualize the
             tree in a PDF
"""

import math

from graphviz import Digraph

from mysklearn import myutils

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        # Set X_train and y_train attributes
        self.X_train = X_train
        self.y_train = y_train

        # Find all possible values of all attributes
        attribute_vals = [[] for i in range(len(X_train[0]))]
        for row in X_train:
            for i, val in enumerate(row):
                if val not in attribute_vals[i]:
                    attribute_vals[i].append(val)

        # Find all possible y values
        y_vals = []
        for val in y_train:
            if val not in y_vals:
                y_vals.append(val)

        # Sort all the possible attribute and y values (to match test cases)
        for val_list in attribute_vals:
            val_list.sort()

        y_vals.sort()

        # Find the frequency distribution of the y values
        y_val_dist = {val: 0 for val in y_vals}
        for val in y_train:
            y_val_dist[val] += 1

        # Make the first recursive call to the TDIDT function and set the tree attribute
        self.tree = self._tdidt_recursive(list(range(len(X_train))), list(range(len(X_train[0]))),
                                          attribute_vals, y_vals, y_val_dist, len(X_train))

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # Verify that the tree has been fitted
        if self.tree is None:
            raise ValueError("Cannot predict without a fitted tree, try running fit() method first")

        # List of generated predictions
        y_pred = []

        # Generate a prediction for every row in X_test
        for X_row in X_test:
            # Store a reference to the current subtree (narrowed as tree traversed)
            subtree = self.tree

            # Iterate until a prediction found
            prediction_found = False
            while not prediction_found:
                # Figure out what kind of node is at the head of the subtree
                match subtree[0]:
                    # Leaf node
                    case "Leaf":
                        # Predict the value in the leaf node
                        prediction_found = True
                        y_pred.append(subtree[1])

                    # Attribute split
                    case "Attribute":
                        # Get the attribute index
                        attribute_idx = int(subtree[1][3])  # Index is 4th character in default name

                        # Get this row's value for the splitting attriute
                        attribute_val = X_row[attribute_idx]

                        # Iterate through all the value branches until a match is found
                        for value_branch in subtree[2:]:
                            # Check if value branch matches row's attribute value
                            if value_branch[1] == attribute_val:
                                # Match found, set subtree to node at end of branch
                                subtree = value_branch[2]

                    # Unexpected node label
                    case _:
                        raise ValueError(f"Unexpected node label encountered: '{subtree[0]}'")

        return y_pred

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        # Verify that tree has been fitted
        if self.tree is None:
            raise ValueError("Cannot print decision rules for unfitted tree, try running fit() method first")
        
        # If tree is just a leaf, print a default rule and return
        if self.tree[0] == "Leaf":
            print(f"{class_name} = \"{self.tree[1]}\"")
            return
        
        # Tree starts with attribute split, make a recursive decision rule function call for every
        # value branch
        # Get the default attribute name
        default_name = self.tree[1]
        
        # Check if attribute names were provided, replacing default if they were
        if attribute_names is None:
            attribute_name = default_name
        else:
            attribute_name = attribute_names[int(default_name[3])]

        # Make a recursive call for every value branch
        for value_branch in self.tree[2:]:
            prefix = f"IF {attribute_name} = {value_branch[1]}"
            self._decision_rule_recursive(value_branch[2], prefix, attribute_names, class_name)

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        # Verify that tree has been fitted already
        if self.tree is None:
            raise ValueError("Cannot visualize an unfitted tree. Try running fit() method first")

        # Instantiate the dot digraph
        dot = Digraph(dot_fname, format="pdf")
        dot.attr(rankdir="TB")

        # Build the graph
        id_counter = -1
        self._dot_visualize_recursive(dot, self.tree, id_counter, attribute_names=attribute_names)

        # Save source and render pdf
        dot.render(filename=dot_fname, outfile=pdf_fname)


    # "private" methods intended for internal use
    def _tdidt_recursive(
            self,
            instances: list[int],
            attributes: list[int],
            attribute_vals: list[list],
            y_vals: list,
            y_val_dist: dict,
            prior_split_size: int
            ) -> list:
        """Recursive function to perform top-down induction of a decision tree.
        
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
            best_tying_y = max(most_common_y_vals, key=lambda y: y_val_dist[y])
            return ["Leaf", best_tying_y, greatest_y_freq, prior_split_size]
        
        # Check if all the remaining instances already belong to the same class (base case)
        y_value_freq = {val: 0 for val in y_vals}
        for i in instances:
            y_value_freq[self.y_train[i]] += 1

        if max(y_value_freq.values()) == sum(y_value_freq.values()):
            # All remaining instances belong to the same class (base case) return remaining class
            best_y = max(y_value_freq.items(), key=lambda t: t[1])[0]
            return ["Leaf", best_y, len(instances), prior_split_size]

        # Iterate across available attributes to find best E_new
        best_new_entropy = math.inf
        best_attribute = -1
        for attribute_idx in attributes:
            # Calculate the new entropy for this attribute
            new_entropy = myutils.calculate_entropy(self.X_train, self.y_train, instances,
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
                                                                      y_val_dist, len(instances))])
        
        # Return the resulting subtree
        return subtree
    
    def _decision_rule_recursive(self, subtree, prefix, attribute_names=None, class_name="class"):
        """Recursive helper for print_decision_rules() method. Prints a full decision rule if at a
        leaf node or makes further recursive calls otherwise
        
        Args:
            subtree (list): The current subtree of the decision tree being traversed
            prefix (str): The decision rule thus far (added on to in further recursive calls)
            attribute_names (optional list): The names of the attributes in the decision tree
                (default names like "att0" used if none provided)
            class_name (str): The name of the class being predicted by decision tree

        Behavior:
            No return but may print a decision rule line
        """
        # Figure out what kind of node is at the head of the subtree
        match subtree[0]:
            # Leaf node (base case, print full decision rule)
            case "Leaf":
                print(prefix, f"THEN {class_name} = \"{subtree[1]}\"")

            # Attribute split
            case "Attribute":
                # Get the default attribute name
                default_name = subtree[1]
                
                # Check if attribute names were provided, replacing default if they were
                if attribute_names is None:
                    attribute_name = default_name
                else:
                    attribute_name = attribute_names[int(default_name[3])]

                # Make a recursive call for every value branch
                for value_branch in subtree[2:]:
                    extended_prefix = f"{prefix} AND {attribute_name} = {value_branch[1]}"
                    self._decision_rule_recursive(value_branch[2], extended_prefix,
                                                  attribute_names, class_name)

            # Unexpected node label
            case _:
                raise ValueError(f"Unexpected node label encountered: '{subtree[0]}'")
            
    def _dot_visualize_recursive(self,
                                 dot: Digraph,
                                 subtree: list,
                                 id_counter: int,
                                 parent_id: int | None=None,
                                 parent_edge_label: str | None=None,
                                 attribute_names: list[str] | None=None
                                 ) -> int:
        """Recursive function to visualize a decision tree using graphviz.
        
        Args:
            dot (Digraph): The graphviz Digraph object being used to visualize the decision tree
            subtree (list of obj): The current subtree in the decision tree visualization traversal
            id_counter (int): The unique integer id of the last node created
            parent_id (optional int): The integer id of the parent node of this one
            parent_edge_label (optional str): The value label for the edge leading from the parent
                node to this one
            attribute names (optional list of str): The list of appropriate labels for each
                attribute (default names like "att0" used if none provided)
        
        Returns:
            id_counter (int): The updated node id counter after generating the subtree headed at
                this node
        """

        # Iterate the id counter for the new node to be created
        id_counter += 1

        # Figure out what kind of node is at the head of the subtree
        match subtree[0]:
            # Leaf node (base case, add leaf to graph and return)
            case "Leaf":
                # Define HTML-like label for size formatting
                label = f"<<B>{subtree[1]}</B><BR/><FONT POINT-SIZE=\"10\">{subtree[2]} / {subtree[3]}</FONT>>"

                # Create new leaf node in the dot graph
                id_counter += 1
                dot.node(f"n{id_counter}", label=label, shape="ellipse")

                # If this node has a parent, add the appropriate edge
                if parent_id is not None:
                    dot.edge(f"n{parent_id}", f"n{id_counter}", label=parent_edge_label)

                return id_counter
            
            # Attribute split
            case "Attribute":
                # Get the default attribute name
                default_name = subtree[1]

                # Check if attribute names were provided, replacing default if they were
                if attribute_names is None:
                    attribute_name = default_name
                else:
                    attribute_name = attribute_names[int(default_name[3])]

                # Add an attribute node to the dot diagram
                dot.node(f"n{id_counter}", label=attribute_name, shape="box", style="rounded")

                # If this node has a parent, add the appropriate edge
                if parent_id is not None:
                    dot.edge(f"n{parent_id}", f"n{id_counter}", label=parent_edge_label)

                # Make a recursive call for every value branch
                this_node_id = id_counter
                for value_branch in subtree[2:]:
                    id_counter = self._dot_visualize_recursive(dot, value_branch[2], id_counter,
                                                               this_node_id, str(value_branch[1]),
                                                               attribute_names)

            # Unexpected node label
            case _:
                raise ValueError(f"Unexpected node label encountered: \"{subtree[0]}\"")

        # Return the id counter, representing the int id of the last node created
        return id_counter


class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        # Re-assign regressor attribute to fitted linear regression line
        self.regressor = MySimpleLinearRegressor()
        self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # Verify that model has been fit (regressor is not none)
        if self.regressor is None:
            raise AttributeError('Regressor has not been set. Try running fit() before predict().')

        # Run linear regressor to get numeric predictions
        numeric_preds = self.regressor.predict(X_test)

        # Return list of all numeric predictions after running through discretizer
        return [self.discretizer(pred) for pred in numeric_preds]
    

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closest neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        # Define both return lists
        distances = []
        indices = []

        # Find k nearest neighbors for all test instances
        for A in X_test:
            # Calculate the distance from A to each training instance
            all_distances = []
            for B in self.X_train if self.X_train is not None else []:  # Assume no training values for null X_train
                all_distances.append(self.euclidean_distance(A, B))

            # Select the top-k shortest distances (as tuples with index and distance)
            top_k_dists = sorted(list(enumerate(all_distances)), key=lambda d: d[1])[:self.n_neighbors]

            # Append indices and distances to return lists
            distances.append([d for _, d in top_k_dists])
            indices.append([i for i, _ in top_k_dists])

        # Return the distances and indices lists
        return distances, indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # Define the list of class predictions
        y_predicted = []

        # Calculate the nearest neighbor indices using the kneighbors function
        _, indices = self.kneighbors(X_test)

        # Make a prediction based on each set of neighbor indices
        for neighbor_set in indices:
            # Find the class (y) of each nearest neighbor
            neighbor_ys = [self.y_train[i] for i in neighbor_set] if self.y_train is not None else []   # Assume no training data for y_train None
            
            # Predict the most common class
            y_freqs = myutils.get_value_frequencies(neighbor_ys)
            y_predicted.append(max(y_freqs, key=lambda y: y_freqs[y]))

        return y_predicted
    
    @staticmethod
    def euclidean_distance(A: list, B: list) -> float:
        """Calculates the euclidean distance between two vectors, represented by lists
        
        Args: The two vectors representing points in a space to calculate the distance between,
        termed A and B

        Returns: The euclidean distance between vectors A and B, as a float
        """
        # Verify that a and b have the same length greater than 0
        if len(A) == 0 or len(B) == 0:
            raise ValueError('Cannot calculate the euclidean distance with an empty list')
        
        if len(A) != len(B):
            raise ValueError('Vectors a and b must have the same number of components')
        
        # Calculate the element-wise difference between the vectors
        differences = [a - b for a, b in zip(A, B)]

        # Calculate the sum of square differences
        sum_squares = 0.0
        for diff in differences:
            sum_squares += diff**2

        # Return the square root of the sum of squares
        return sum_squares**0.5


class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        # Calculate the frequency of each label in y_train
        y_freqs = myutils.get_value_frequencies(y_train)

        # Set most common label to the y with the greatest frequency
        self.most_common_label = max(y_freqs, key=lambda y: y_freqs[y])

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # Return the most common label for each test instance
        return [self.most_common_label] * len(X_test)


class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(dict of object to float): The prior probabilities computed for each
            label in the training set.
        conditionals(dict of (int, object) tuple to dict of object to float): The conditional probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.conditionals = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the conditional probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and conditionals.
        """
        # Verify X_train and y_train are the same length
        if len(X_train) != len(y_train):
            raise ValueError("Parameters X_train and y_train must have the same length")
        
        # Calculate prior probabilities
        self.priors = {}

        # Find the number of occurrences for each value in y_train
        y_freqs = myutils.get_value_frequencies(y_train)

        # Divide each occurrence count by length of y_train to get priors
        for y_val, freq in y_freqs.items():
            self.priors[y_val] = freq / len(y_train)

        # Calculate conditional probabilities
        self.conditionals = {}

        # Iterate through predictor variables by index
        for predictor_idx in range(len(X_train[0])):
            # Extract a list of all values of that predictor
            predictor_vals = [x[predictor_idx] for x in X_train]

            # Find all unique predictor values
            unique_predictor_vals = list(set(predictor_vals))

            # Find all conditional probs for those predictor vals given y values
            for predictor in unique_predictor_vals:
                # Instantiate new sub-dict for this predictor value
                self.conditionals[(predictor_idx, predictor)] = {}

                for y_val, freq in y_freqs.items():
                    # Count all matches of target predictor and y value
                    predictor_y_matches = sum(1 if pred == predictor and y == y_val else 0 for pred, y in zip(predictor_vals, y_train))

                    # Add conditional probability as number of matches divided by number of y val occurrences
                    self.conditionals[(predictor_idx, predictor)][y_val] = predictor_y_matches / freq

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # Ensure model was fitted in advance
        if self.priors is None or self.conditionals is None:
            raise RuntimeError("Cannot make predictions with an unfitted model")

        # Define list for predicted values
        y_predicted = []

        # Iterate through each test instance in X_test
        for x_instance in X_test:
            # Define a dict for each y value's probability
            y_value_probs = {}

            # Iterate through each possible y value
            for y_val, prior_prob in self.priors.items():
                y_val_prob = prior_prob

                # Iterate through each predictor variable with index
                for predictor_idx, predictor in enumerate(x_instance):
                    # Multiply the probability for the y value by the corresponding conditional prob
                    y_val_prob *= self.conditionals[(predictor_idx, predictor)][y_val]

                # Save the joint probability for the y value
                y_value_probs[y_val] = y_val_prob

            # Make a prediction based on the y value probabilities dict
            y_predicted.append(max(y_value_probs.items(), key=lambda x: x[1])[0])

        return y_predicted

