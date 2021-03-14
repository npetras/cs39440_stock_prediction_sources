"""
Module deals with the loading of data
"""

import os
import pandas as pd


def from_csv(rel_path):
    """
    Reads data from a CSV file using the Pandas library
    :return: Pandas data frame of the CSV file loaded
    """
    djia_abs_path = os.path.abspath(rel_path)
    return pd.read_csv(djia_abs_path)
