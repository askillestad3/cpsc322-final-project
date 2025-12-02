"""
utility functions for mysklearn package
"""

import math
import numpy as np
import matplotlib.pyplot as plt

def my_euclidean_distance(point1, point2):
    """compute euclidean distance"""
    diff = np.array(point1) - np.array(point2)
    return float(np.sqrt(np.sum(diff ** 2)))


def get_value_frequencies(value_list: list) -> dict:
    """Helper function to find the frequencies of all distinct values in a list
    
    Args:
        value_list: The list of all values whose frequencies are to be calculates
        
    Returns: A dictionary mapping each distinct value to an integer indicating its frequency

    Notes:
        - The values in value_list must be hashable (typically immutable)
    """
    frequency_dict = {}
    for val in value_list:
        if val in frequency_dict:
            frequency_dict[val] += 1
        else:
            frequency_dict[val] = 1
    return frequency_dict


def calculate_entropy(
        X: list[list],
        y: list,
        instances: list[int],
        attribute_idx: int,
        attribute_vals: list,
        y_vals: list
        ) -> float:
    """Calculates the new entropy after splitting on a given attribute.

    Args:
        X (list of list of obj): The predictor matrix (presumably for a decision tree)
        y (list of obj): The response vector (as above)
        instances (list of int): The indexes of the instances to use in calculation
        attribute_idx (int): The index of the given attribute in each row of the predictor matrix X
        attribute_vals (list of obj): The list of all distinct values in the specified attribute
        y_vals (list of obj): The list of all distinct values in the response vector y

    Returns:
        entropy (float): The new entropy value (E_new) after splitting on the specified attribute
    """
    
    # Find the frequencies of the attribute values provided
    attribute_val_freq = {val: 0 for val in attribute_vals}
    for i in instances:
        attribute_val = X[i][attribute_idx]
        attribute_val_freq[attribute_val] += 1

    # Start entropy at 0
    entropy = 0.0

    # Calculate the entropy component for each distinct attribute value provided
    for attribute_val, attribute_freq in attribute_val_freq.items():
        # Dict to count occurrences of each y value for this attribute value
        y_value_freq = {val: 0 for val in y_vals}

        # Iterate across instances to count y value occurrences
        for i in instances:
            # Only use rows with this attribute value
            if X[i][attribute_idx] != attribute_val:
                continue

            # Add 1 to the frequency count of corresponding y value
            y_value_freq[y[i]] += 1

        # Calculate this entropy component
        entropy_component = sum(0 if y_freq == 0 
                                else -y_freq / attribute_freq * math.log(y_freq / attribute_freq, 2)
                                for y_freq in y_value_freq.values())
        entropy_component *= attribute_freq / len(instances)

        # Add this attribute value's component to the total entropy
        entropy += entropy_component

    return entropy

def plot_frequency_diagram(table, col_name):
    """Create a frequency bar chart for a categorical column"""
    values = table.get_column(col_name, include_missing_values=False)
    freq = {}
    for v in values:
        freq[v] = freq.get(v, 0) + 1
    plt.bar(freq.keys(), freq.values()) # type: ignore
    plt.xlabel(col_name)
    plt.ylabel("Frequency")
    plt.title(f"Frequency of {col_name}")
    plt.show()

def plot_histogram(table, col_name, bins=10):
    """Create a histogram for a numeric column"""
    values = table.get_column(col_name, include_missing_values=False)
    plt.hist(values, bins=bins)
    plt.xlabel(col_name)
    plt.ylabel("Count")
    plt.title(f"Histogram of {col_name}")
    plt.show()

def plot_scatter(table, x_col, y_col):
    """Create a scatter plot between two numeric columns"""
    x = table.get_column(x_col, include_missing_values=False)
    y = table.get_column(y_col, include_missing_values=False)
    plt.scatter(x, y)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col}")
    plt.show()

def plot_box_whisker_by_category(table, category_col, value_col):
    """Creates a box-whisker plot for a numeric column grouped by categories"""
    categories = sorted(list(set(table.get_column(category_col))))
    data = []
    for cat in categories:
        cat_vals = [
            row[table.column_names.index(value_col)]
            for row in table.data
            if row[table.column_names.index(category_col)] == cat
            and row [table.column_names.index(value_col)] != "NA"
        ]
        data.append(cat_vals)
    plt.boxplot(data, label=categories)
    plt.xlabel(category_col)
    plt.ylabel(value_col)
    plt.title(f"{value_col} distribution by {category_col}")
    plt.show()
