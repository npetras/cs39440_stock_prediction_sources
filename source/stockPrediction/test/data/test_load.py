"""
Test module for main.data.load
"""

import unittest
import os.path
import pandas as pd
from main.data import load



class TestLoad(unittest.TestCase):
    """
    Test class for main.data.load
    """
    def test_from_csv(self):
        """
        Provides the DJIA dataset to be loaded, provides the absolute path, based on the relative
        path that this file is contained within.
        :return:
        """
        curr_filepath = os.path.abspath(__file__)
        djia_filepath = os.path.join(curr_filepath,
                                     '../../../../../datasets/existing/dow_jones/Combined_News_DJIA'
                                     '.csv')
        abs_djia_filepath = os.path.abspath(djia_filepath)
        result = load.from_csv(abs_djia_filepath)
        self.assertEqual(isinstance(result, pd.DataFrame), True)


if __name__ == '__main__':
    unittest.main()
