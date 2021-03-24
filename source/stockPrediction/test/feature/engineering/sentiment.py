"""
Test module for main.feature.engineering.sentiment
"""

import unittest
import re
from main.feature.engineering import sentiment


class TestSentiment(unittest.TestCase):
    """
    Test class for module main.feature.engineering.sentiment
    """
    def test_compound_score(self):
        """
        Checks if the compound_score method, outputs the score in the correct format. The score
        itself is not checked since that may change, if VADER is updated.
        """
        text = 'Russia Today: Columns of troops roll into South Ossetia; footage from ' \
               'fighting (YouTube)'
        regex = re.compile(r'-\d\.\d+')
        result = str(sentiment.compound_score(text))
        self.assertTrue(re.match(regex, result))


if __name__ == '__main__':
    unittest.main()
