"""
This program will perform data preparation
and exploratory data analysis (EDA) of a real-world auto dataset.
"""

import copy
import csv

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
        if isinstance(col_identifier, int):
            col_index = col_identifier
            if col_index < 0 or col_index >= len(self.column_names):
                raise ValueError("Invalid column index")
        elif isinstance(col_identifier, str):
            if col_identifier not in self.column_names:
                raise ValueError("Invalid column name")
            col_index = self.column_names.index(col_identifier)
        else:
            raise ValueError("Invalid column identifier")

        col = []
        for row in self.data:
            val = row[col_index]
            if not include_missing_values and val == "NA":
                continue
            col.append(val)
        return col

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leaves values as-is that cannot be converted to numeric.
        """
        for i, row in enumerate(self.data):
            for j, val in enumerate(row):
                try:
                    self.data[i][j] = float(val)
                except (ValueError, TypeError):
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Parameters:
            row_indexes_to_drop (list of int): list of row indexes to remove from the table data.
        """
        for index in sorted(row_indexes_to_drop, reverse=True):
            if 0 <= index < len(self.data):
                self.data.pop(index)

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
        with open(filename, "r", encoding="utf-8") as infile:
            reader = csv.reader(infile)
            self.column_names = next(reader)
            self.data = list(reader)
        self.convert_to_numeric()
        return self

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
        """
        stats = []
        for col_name in col_names:
            col_index = self.column_names.index(col_name)
            values = [row[col_index] for row in self.data if row[col_index] != "NA"]
            if not values:
                continue
            min_val = min(values)
            max_val = max(values)
            mid_val = (min_val + max_val) / 2
            avg_val = sum(values) / len(values)

            sorted_vals = sorted(values)
            n = len(sorted_vals)
            if n % 2 == 1:
                median_val = sorted_vals[n // 2]
            else:
                median_val = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2

            stats.append([col_name, min_val, max_val, mid_val, avg_val, median_val])
        return MyPyTable(["attribute", "min", "max", "mid", "avg", "median"], stats)
