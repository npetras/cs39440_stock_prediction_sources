"""
Test module for main.modelling.run
"""

import unittest
from main.modelling import run


class TestRun(unittest.TestCase):
    """
    Test class for main.modelling.run
    """

    def test_with_vectorizer_error(self):
        """
        Expects to produce an exception when both the stemming and lemmatization parameters are
        set to true
        :return:
        """
        self.assertRaises(Exception, run.with_vectorizer,
                          'Cannot use both stemming and lemmatization')


if __name__ == '__main__':
    unittest.main()
