"""
Module for main.data.manipulate
"""
import unittest
from main.data import manipulate


class TestManipulate(unittest.TestCase):
    """
    Test class for main.data.manipulate functions
    """

    def test_combine_string(self):
        """
        Provides a list a basic list of strings to combine_string()
        """
        string_list = ['hello how', 'are', 'you doing', 'today?']
        expected_string = 'hello how are you doing today?'
        resulting_string = manipulate.combine_strings(string_list)
        self.assertEqual(expected_string, resulting_string)

    def test_treebank_detokenize_basic(self):
        """
        Provides a list of expected tokens, if the sklearn_tokenize() was run on a piece of text.
        """
        tokenized_text = [
            'Hello', 'didn', 'see', 'you', 'at', 'the', 'talk', 'were', 'you',
            'there'
        ]
        detokenized_text = 'Hello didn see you at the talk were you there'
        result = manipulate.treebank_detokenize(tokenized_text)
        self.assertEqual(result, detokenized_text)

    def test_detokenize_headline(self):
        """
        Provides a tokenized headline from the DJIA dataset, as if it was tokenized by
        sklearn_tokenize()
        :return:
        """
        tokenized_text = [
            'Rice', 'Gives', 'Green', 'Light', 'for', 'Israel', 'to', 'Attack',
            'Iran', 'Says', 'has', 'no', 'veto', 'over', 'Israeli', 'military',
            'ops'
        ]
        detokenized_text = 'Rice Gives Green Light for Israel to Attack Iran Says has no veto ' \
                           'over Israeli military ops'
        result = manipulate.treebank_detokenize(tokenized_text)
        self.assertEqual(result, detokenized_text)


if __name__ == '__main__':
    unittest.main()
