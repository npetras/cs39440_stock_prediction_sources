import unittest
from main.classifier import run


class TestRun(unittest.TestCase):

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
