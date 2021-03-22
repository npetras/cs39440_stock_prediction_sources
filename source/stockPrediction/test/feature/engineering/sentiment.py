import unittest
from main.feature.engineering import sentiment
import re

class MyTestCase(unittest.TestCase):
    def test_compound_score(self):
        text = 'Russia Today: Columns of troops roll into South Ossetia; footage from ' \
               'fighting (YouTube)'
        regex = re.compile(r'-\d\.\d+')
        result = str(sentiment.compound_score(text))
        self.assertTrue(re.match(regex, result))


if __name__ == '__main__':
    unittest.main()
