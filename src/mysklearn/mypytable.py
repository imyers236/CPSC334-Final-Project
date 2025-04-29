##############################################
# Programmer: Ian Myers
# Class: CPSC 322-02, Fall 2024
# Programming Assignment #7
# 11/11/24
# Description: This program creates an object called
#   mypytable with function that affect data in many ways
##############################################

import copy
import csv
from tabulate import tabulate
from mysklearn import myutils

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def display_rows_at_indexes(self, indexes):
        """Prints the desired rows in a nicely formatted grid structure.
        """
        sub_table = []
        for i in indexes:
            sub_table.append(self.data[i])
        print(tabulate(sub_table, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)
    
    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        # Checks type of col_identifier
        if(type(col_identifier) != int and type(col_identifier) != str):
            raise ValueError(col_identifier)
        elif type(col_identifier) == str:
            # Finds the index if the value is a string
            col_identifier = self.column_names.index(col_identifier)
        col = []
        # Fills in the new list
        if include_missing_values:
            for row in self.data:
                col.append(row[col_identifier])
        else:
            for row in self.data:
                if row[col_identifier] != "NA":
                    col.append(row[col_identifier])
        return col

    def get_columns(self, col_identifier_list, include_missing_values=True):
        """Extracts columns from the table data as a list.

        Args:
            col_identifier_list(list of str or int): list of strings for a column name or ints
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        rows = []
        for col in col_identifier_list:
            rows.append(self.get_column(col, include_missing_values))
        columns = self.column_to_row(rows)
        return columns
    
    def get_mins(self, list):
        """
        Finds the mins for each column name
        """
        mins = []
        for i in list:
            mins.append(min(self.get_column(i)))
        return mins
    
    def get_maxes(self, list):
        """
        Finds the mins for each column name
        """
        maxes = []
        for i in list:
            maxes.append(max(self.get_column(i)))
        return maxes

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                try:
                    float(self.data[i][j])
                except ValueError:
                    pass
                else:
                    self.data[i][j] = float(self.data[i][j])

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        # creates a copy of the table will all the original indexes
        old_table = self.data.copy()
        for i in range(len(self.data)):
            if i in row_indexes_to_drop:
                # finds the new index to be deleted
                shifted_index = self.data.index(old_table[i])
                del self.data[shifted_index]

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        table = []
        # 1. opens the file
        with open(filename, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                table.append(row)
        # 2. process the file
        self.column_names = table[0]
        self.data = table[1:]
        self.convert_to_numeric()
        # 3. close the file
        file.close()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(self.column_names)
            writer.writerows(self.data)
        file.close() 

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        # list to return indexes of dups
        dups = []
        # dictionary to hold original values and refer back to
        key_dict = {}
        key_col_indexes = []
        # Finds the indexes of the columns that must be checked for duplicates
        for col in key_column_names:
            key_col_indexes.append(self.column_names.index(col))
        
        # Adds the values of the row key to one string and check it against the dict
        for i in range(len(self.data)):
            key_str = ""
            for j in key_col_indexes:
                key_str += str(self.data[i][j])
            if key_str in key_dict.keys():
                dups.append(i)
            else:
                key_dict[key_str] = 1


        return dups
    
    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        i = 0
        while i < len(self.data):
            for j in range(len(self.data[i])):
                if self.data[i][j] == "NA":
                    del self.data[i]
                    # Resets index to shifted table and breaks from loop
                    i = i -1
                    break
            i += 1

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        avg_col = self.get_column(col_name,False)
        avg_val = sum(avg_col) / len(avg_col)
        col_index = self.column_names.index(col_name)
        for i in range(len(self.data)):
            if self.data[i][col_index] == "NA":
                self.data[i][col_index] = avg_val

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values should in the columns to compute summary stats
                for should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        header = ["attribute", "min", "max", "mid", "avg", "median"]
        stats_table = []
        for col in col_names:
            # gets column
            col_data = self.get_column(col,False)
            # checks if col is empty
            if col_data == []:
                break
            # performs the needed stats
            col_min = min(col_data)
            col_max = max(col_data)
            col_mid = (col_max + col_min) / 2
            col_avg = sum(col_data) / len(col_data)
            col_data.sort()
            # finds if col is odd the returns middle val or avg of two middle points
            if len(col_data) % 2 == 1:
                col_median = col_data[(len(col_data) // 2)]
            else:
                col_median = (col_data[((len(col_data) // 2))] + col_data[((len(col_data) // 2)) - 1]) / 2
            stats_table.append([col,col_min,col_max,col_mid,col_avg,col_median])
        return MyPyTable(header,stats_table) # TODO: fix this

    def join_header(self, other_table):
        """
        Returns:
            the new header of the join tables, the indexes of the left table that
                the right table does not have and the indexes of the right table that
                the left table does not have 
        """
        new_header = copy.deepcopy(self.column_names)
        left_unique_attributes = []
        right_unique_attributes = []
        # fills up header and right unique attributes
        for i,col in enumerate(other_table.column_names):
            if col not in new_header:
               right_unique_attributes.append(i)
               new_header.append(col)

        # fills up the left unique attributes
        for i,col in enumerate(self.column_names):
            if col not in other_table.column_names:
               left_unique_attributes.append(i)
        
        return new_header, left_unique_attributes, right_unique_attributes
    
    def get_key_columns(self, other_table, key_column_names):
        """
        Returns:
            two 2D arrays one for each table with only the values of the key columns
                in the rows
        """
        key_columns_left = []
        key_columns_right = []
        for col in key_column_names:
            key_columns_left.append(self.get_column(col, False))
            key_columns_right.append(other_table.get_column(col, False))
        # converts table from col by row to row by col
        key_columns_left = self.column_to_row(key_columns_left)
        key_columns_right = self.column_to_row(key_columns_right)
        return key_columns_left, key_columns_right

    def column_to_row(self,table):
        """
        Returns:
            new table by changing the indexes from column X row to row X column

        Args:
            self
            table that needs to be changed
        """
        new_table = [[0] * len(table) for i in range(len(table[0]))]
        for j in range(len(table)):
            for i in range(len(table[j])):
                new_table[i][j] = table[j][i]
        return new_table
    
    def find_matches_and_nons(self, key_columns_left, key_columns_right):
        """
        Returns:
            a dict of the matches with the indexes of in each table and the 
                non matches and their indexes
        """
        matches_dict = {}
        left_nons = []
        right_nons = []

        # finds where there are matches
        for i,row_left in enumerate(key_columns_left):
            for j,row_right in enumerate(key_columns_right):
                if row_left == row_right:
                    if i not in matches_dict:
                        matches_dict[i] = [j]
                    else:
                        matches_dict[i].append(j)
            if i not in matches_dict:
                left_nons.append(i)
        j = 0
        while j < len(key_columns_right):
            exist_match = False
            for match_list in matches_dict.values():
                if j in match_list:
                    exist_match = True
            if not exist_match:
                right_nons.append(j)
            j += 1
        return matches_dict, left_nons, right_nons
                    
    
    
    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        new_data = []
        matches_dict = {}
        # creates new header with the appropriate attributes
        new_header = copy.deepcopy(self.column_names)
        right_unique_attributes = []
        # fills up header and right unique attributes
        for i,col in enumerate(other_table.column_names):
            if col not in new_header:
               right_unique_attributes.append(i)
               new_header.append(col)

        key_columns_left, key_columns_right = self.get_key_columns(other_table, key_column_names)

        # finds where there are matches
        for i,row_left in enumerate(key_columns_left):
            for j,row_right in enumerate(key_columns_right):
                if row_left == row_right:
                    if i not in matches_dict:
                        matches_dict[i] = [j]
                    else:
                        matches_dict[i].append(j)
                    
        
        # merges the matching rows in a new table
        for left_row in matches_dict:
            right_att_only = []
            for match in matches_dict[left_row]:
                right_att_only = []
                for right_col_index in right_unique_attributes:
                    right_att_only.append(other_table.data[match][right_col_index])
                new_data.append(self.data[left_row]+right_att_only)
        return MyPyTable(new_header,new_data) 

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        new_data = []
        new_header, left_unique_attributes, right_unique_attributes = self.join_header(other_table)
        key_columns_left, key_columns_right = self.get_key_columns(other_table, key_column_names)
        matches_dict, left_nons, right_nons = self.find_matches_and_nons(key_columns_left, key_columns_right)

        # completes the inner join
        for left_row in matches_dict:
            right_att_only = []
            for match in matches_dict[left_row]:
                right_att_only = []
                for right_col_index in right_unique_attributes:
                    right_att_only.append(other_table.data[match][right_col_index])
                new_data.append(self.data[left_row]+right_att_only)
        
        # adds in the left non-matches with NA filled in
        for row_index in left_nons:
            row_left_with_NA = self.data[row_index].copy()
            for val in right_unique_attributes:
                row_left_with_NA.append("NA")
            new_data.append(row_left_with_NA)
        # adds in the right non-matches with NA filled in
        for row_index in right_nons:
            row_right_with_NA = [None] * len(new_header)
            # Fills the non NA values in the correct index
            for i,col in enumerate(other_table.column_names):
                index = new_header.index(col)
                row_right_with_NA[index] = other_table.data[row_index][i]
            # Fills the NAs in the correct index
            for val in left_unique_attributes:
                row_right_with_NA[val] = "NA"
            new_data.append(row_right_with_NA)

        return MyPyTable(new_header, new_data) # TODO: fix this


