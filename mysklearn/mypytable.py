
"""
Programmer: Miles Mercer (skeleton code by Dr. Luitel)
Class: CPSC 322, Fall 2025
Programming Assignment #4
10/24/25
Description: This file implements an example data table with a number of useful
             member functions for data analysis, including column extraction,
             data cleaning, and joining
"""

import copy
import csv
from tabulate import tabulate

import numpy as np

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names (list of str): M column names
        data (list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Parameters:
            column_names (list of str): initial M column names (None if empty)
            data (list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)


    def copy(self):
        """Copy constructor for MyPyTable
        
        Returns: A new MyPyTable object identical to this one
        """
        return MyPyTable(self.column_names, self.data)
        

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure."""
        print(tabulate(self.data, headers=self.column_names))


    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            tuple: (N, M) where N is number of rows and M is number of columns
        """
        # Cover base case where data is an empty string, representing a 0 by 0 empty datatable
        if len(self.data) == 0:
            return 0, 0

        # Data table contains at least one list, return dimensions
        if not isinstance(self.data[0], list):
            raise ValueError(f"Expected self.data to be a list of lists. Encountered {type(self.data[0])} type instead.")
        
        return len(self.data), len(self.data[0])
        

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Parameters:
            col_identifier (str or int): string for a column name or int
                for a column index
            include_missing_values (bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Raises:
            ValueError: if col_identifier is invalid
        """
        # Identify the index of the column identifier in the column names list
        if not col_identifier in self.column_names:
            raise ValueError(f"Column identifier '{col_identifier}' is not in the column names list.")

        column_index = self.column_names.index(col_identifier)

        # Collect all column values and return
        column_values = []

        for row in self.data:
            if include_missing_values or row[column_index] != "NA":
                column_values.append(row[column_index])

        return column_values


    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leaves values as-is that cannot be converted to numeric.
        """
        # For each cell in table, try to convert to a float, but continue if unsuccessful
        for i in range(len(self.data)):
            for j, cell in enumerate(self.data[i]):
                try:
                    self.data[i][j] = float(cell)
                except:
                    continue
        

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Parameters:
            row_indexes_to_drop (list of int): list of row indexes to remove from the table data.
        """
        # Use list comprehension to form a list of rows whose indexes are not in the parameter drop
        # index list and assign to self.data
        self.data = [row for i, row in enumerate(self.data) if i not in row_indexes_to_drop]


    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: returns self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Uses the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load.
        """
        # Open the specified csv file
        with open(filename, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)

            # Grab the header row
            self.column_names = next(csv_reader)

            # Iterate through the rest of the file to get table data
            self.data = []

            for row in csv_reader:
                self.data.append(row)

        # Convert all numeric values in table to floats
        self.convert_to_numeric()

        return self
    

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to save the contents to.

        Notes:
            Uses the csv module.
        """
        # Open the specified csv file
        with open(filename, 'w') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Save the header row
            csv_writer.writerow(self.column_names)

            # Save the table data
            csv_writer.writerows(self.data)


    def find_duplicates(self, key_column_names=None):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Parameters:
            key_column_names (list of str): column names to use as row keys.

        Returns:
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
            The first instance of a row is not considered a duplicate.
        """
        # If no key columns are given, use all columns
        if key_column_names is None:
            key_column_names = self.column_names

        # Confirm that all key column names provided are in the column names list
        for name in key_column_names:
            if name not in self.column_names:
                raise ValueError(f"Provided key column name '{name}' is not a column name in the table.")
            
        # Find indexes of all key column names
        key_column_indexes = [i for i, name in enumerate(self.column_names) if name in key_column_names]    # Ignores original order of key column names

        # Instantiate an empty set of tuples for unique key column value combinations
        unique_key_combos = set()

        # Instantiate list to track duplicate row indexes
        duplicate_row_indexes = []

        # For each row, form a tuple from key column values and check if it's in unique key combo list, recording index if so
        for i, row in enumerate(self.data):
            key_column_values = tuple([row[index] for index in key_column_indexes])

            if key_column_values in unique_key_combos:
                duplicate_row_indexes.append(i)             # Row is a duplicate, record row index
            else:
                unique_key_combos.add(key_column_values)    # Row is unique, add to unique key combo set

        return duplicate_row_indexes
    

    def drop_duplicates(self, key_column_names=None):
        """Removes all duplicate rows based on a provided list of key columns (or all columns if no
        list given)

        Parameters:
            - key_column_names: The list of attributes used to identify rows (i.e., other attributes
              may differ while the row is still considered a duplicate)

        Notes:
            - Duplicate rows are dropped in-place, rather than returning a new table without the
              duplicates
        """
        # Find the indexes of the duplicate rows
        duplicate_row_indexes = self.find_duplicates(key_column_names)

        # Reconstruct data table to exclude identified duplicates
        self.drop_rows(duplicate_row_indexes)
    

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA")."""
        # Use list comprehension to form a list of rows with no missing values and assign to self.data
        self.data = [row for row in self.data if "NA" not in row]


    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
        by the column's original average.

        Parameters:
            col_name (str): name of column to fill with the original average (of the column).
        """
        # If data table is empty, return because there can be no missing values
        if len(self.data) == 0:
            return
        
        # Verify provided column name is in table column names
        if col_name not in self.column_names:
            raise ValueError(f"Specified column name '{col_name}' is not in table column names.")

        # Verify specified column is a numeric column
        col_index = self.column_names.index(col_name)
        for row in self.data:
            # Skip missing values to find a present value to evaluate
            if row[col_index] == "NA":
                continue

            # Non-missing value found, verify that it is a numeric type
            if type(self.data[0][col_index]) not in [int, float, bool]:     # Bool considered numeric because True and False are 1 and 0
                raise ValueError(f"Specified column name '{col_name}' is not a numeric column. Maybe call convert_to_numeric first?")
            else:
                break

        # Collect all non-missing values from specified column
        column_values = self.get_column(col_name, include_missing_values=False)

        # Verify specified column actually contains non-missing values
        if len(column_values) == 0:
            raise ValueError(f"Specified column '{col_name}' contains no non-missing values. Average value cannot be computed")

        # Calculate average of non-missing values in specified column
        avg_column_val = sum(column_values) / len(column_values)

        # Replace all missing values ("NA") in specified column with the calculated average value
        for i, row in enumerate(self.data):
            if row[col_index] == "NA":
                self.data[i][col_index] = avg_column_val
   

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Parameters:
            col_names (list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values in the columns to compute summary stats
            should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
            Columns with no non-missing values will not be included in the returned table
        """
        # Ensure all provided column names are in the table's column names
        for name in col_names:
            if name not in self.column_names:
                raise ValueError(f"Provided column name '{name}' is not a column name in the table.")

        # Instantiate an empty summary table to return
        summary_table = MyPyTable(column_names=['attribute', 'min', 'max', 'mid', 'avg', 'median'])

        # Iterate through parameter column names
        for name in col_names:
            # Get all non-missing values for the specified column
            column_vals = self.get_column(name, include_missing_values=False)

            # If list is empty, column contains no non-missing values, so skip it
            if len(column_vals) == 0:
                continue

            # Sort the column values
            column_vals.sort()

            # Find the summary statistics for the column
            col_min = column_vals[0]    # List is already sorted, so min at index 0 and max at index -1
            col_max = column_vals[-1]
            col_mid = (col_min + col_max) / 2
            col_avg = sum(column_vals) / len(column_vals)

            if len(column_vals) % 2 == 0:   # Even number of column values
                col_median = (column_vals[len(column_vals) // 2 - 1] + column_vals[len(column_vals) // 2]) / 2
            else:   # Odd number of column values
                col_median = column_vals[len(column_vals) // 2]

            # Add new row with column name and summary statistics
            summary_table.data.append([name, col_min, col_max, col_mid, col_avg, col_median])

        # Return final summary table
        return summary_table


    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
        with other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        # Ensure all provided key column names are contained in this and other table
        for name in key_column_names:
            if name not in self.column_names or name not in other_table.column_names:
                raise ValueError(f"Provided key column name '{name}' is not contained in both joining tables.")

        # Find the index pairs for each column in this and other table
        index_pairs = []
        for name in key_column_names:
            index_pairs.append((self.column_names.index(name), other_table.column_names.index(name)))

        # Instantiate new table with all attributes from this table and all non-key attributes from other table
        join_table_columns = self.column_names.copy()
        join_table_columns.extend([name for name in other_table.column_names if name not in key_column_names])
        join_table = MyPyTable(column_names=join_table_columns)

        # Iterate cartesian product of tables, appending all matching rows to join table
        for my_row in self.data:
            for other_row in other_table.data:
                # Check if rows match on key columns
                rows_match = True
                for my_index, other_index in index_pairs:
                    if my_row[my_index] != other_row[other_index]:
                        rows_match = False
                        break
                
                # If rows don't match, move on to next row combination
                if not rows_match:
                    continue

                # Rows match, combine rows and add to join table
                combined_row = my_row.copy()
                combined_row.extend([val for i, val in enumerate(other_row) if i not in [pair[1] for pair in index_pairs]])
                join_table.data.append(combined_row)

        # Return join table
        return join_table


    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
        other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pads attributes with missing values with "NA".
        """
        # Ensure all provided key column names are contained in this and other table
        for name in key_column_names:
            if name not in self.column_names or name not in other_table.column_names:
                raise ValueError(f"Provided key column name '{name}' is not contained in both joining tables.")

        # Find the index pairs for each column in this and other table
        index_pairs = []
        for name in key_column_names:
            index_pairs.append((self.column_names.index(name), other_table.column_names.index(name)))

        # Instantiate new table with all attributes from this table and all non-key attributes from other table
        join_table_columns = self.column_names.copy()
        join_table_columns.extend([name for name in other_table.column_names if name not in key_column_names])
        join_table = MyPyTable(column_names=join_table_columns)

        # Create a set of tuples for all matched key column value combinations, to help find unmatched rows
        matched_key_combos = set()

        # Iterate cartesian product of tables, appending all matching rows to join table
        for my_row in self.data:
            # Record whether a match is found for this row, in case unmatched row needed for outer join
            match_found = False

            for other_row in other_table.data:
                # Track matching key column values to record match in set
                matched_key_vals = []

                # Check if rows match on key columns
                rows_match = True
                for my_index, other_index in index_pairs:
                    if my_row[my_index] == other_row[other_index]:  # Rows match, record matching value
                        matched_key_vals.append(my_row[my_index])
                    else:   # Rows don't match, lower rows match flag and stop checking key columns
                        rows_match = False
                        break
                
                # If rows don't match, move on to next row combination
                if not rows_match:
                    continue

                # Rows match, combine rows and add to join table
                combined_row = my_row.copy()
                combined_row.extend([val for i, val in enumerate(other_row) if i not in [pair[1] for pair in index_pairs]])
                join_table.data.append(combined_row)

                # Raise match found flag to avoid forming non-matching row for outer join
                match_found = True

                # Record the matching key value combination
                matched_key_combos.add(tuple(matched_key_vals))

            # If no match was found for my row, create a non-matching row and add to outer join table
            if not match_found:
                non_matching_row = my_row.copy()
                non_matching_row.extend(['NA'] * (len(other_table.column_names) - len(key_column_names)))
                join_table.data.append(non_matching_row)

        # Iterate through rows in other table, looking for non-matched key value combinations
        for row in other_table.data:
            key_vals = tuple([row[i] for _, i in index_pairs])
            if key_vals not in matched_key_combos:  # Row unmatched, form non-matching row
                # Begin with NA's for all columns corresponding to this table (key columns will be overwritten)
                non_matching_row = ['NA'] * len(self.column_names)

                # Fill in all key column values
                for my_index, other_index in index_pairs:
                    non_matching_row[my_index] = row[other_index]

                # Extend non-matching row with non-key values from other table
                non_matching_row.extend([val for i, val in enumerate(row) if i not in [pair[1] for pair in index_pairs]])

                # Append non-matching row to full outer join table
                join_table.data.append(non_matching_row)

        # Return join table
        return join_table
    
    def train_test_split(self, num_test_instances: int, seed=None):
        """Performs a train-test split on the data table based on a specified number of test
        instances
        
        Args:
            - num_test_instances: The number of instances desired in the test set
            - seed: Optional seed for Numpy random module

        Returns:
            - train: A MyPyTable object with the training data
            - test: A MyPyTable object with the test data
        """
        # If the number of test instances is equal to or longer than this table, give the full table
        # copy as the test partition
        if num_test_instances >= self.get_shape()[0]:
            return MyPyTable(column_names=self.column_names), self.copy()
        
        # Select a random set of test indices
        if seed is not None:
            np.random.seed(seed)

        test_indices = np.random.choice(list(range(self.get_shape()[0])), num_test_instances, replace=False)

        # Sort the test indices to get test instances in order
        test_indices.sort()

        # Select the train table
        train = self.copy()
        train.drop_rows(test_indices)

        # Select the test table
        test = MyPyTable(column_names=self.column_names)
        test.data = [self.data[i] for i in test_indices]

        # Return train and test partitions
        return train, test
    

    def normalize_column(self, column_name: str):
        """This function takes in a numeric column name and generates a new column with the Z-score
        (number of standard deviations away from mean) for every value. Used for scaling attributes
        before using artificial intelligence algorithms.
        
        Args:
            - column_name: The name of the column to normalize

        Notes:
            - The normalized value column is named <column_name>_NORM
        """
        # Verify that column_name is a column in the table
        if not column_name in self.column_names:
            raise ValueError(f"Column name {column_name} not contained in this table")
        
        # Grab all values for the column
        column_vals = self.get_column(column_name)

        # Calculate the mean and standard deviation for the column
        mean = np.mean(column_vals)
        std_dev = np.std(column_vals)

        # Make a new column with the z-score for each value in column
        self.column_names.append(column_name + '_NORM')
        for i, val in enumerate(column_vals):
            self.data[i].append(float((val - mean) / std_dev))

