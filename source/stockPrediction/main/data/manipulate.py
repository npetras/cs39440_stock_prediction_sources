"""
Functions to manipulate the data in some basic form e.g. combining multiple strings into one
"""

from nltk.tokenize import treebank


def combine_strings(string_list):
    """
    Combines a list of strings into a single string
    :param string_list: A list of strings to be combined
    :return: string_list combined into one string
    """
    combined_string = ''
    for string in string_list:
        combined_string += str(string) + ' '
    return combined_string.strip()


def treebank_detokenize(tokenized_text):
    """
    Uses the TreebankDetokenizer to detokenize the tokenized_list into a string.
    :param tokenized_text: list of tokens to be detokokenized
    :return: string formed from the tokenized_list
    """
    detokenizer = treebank.TreebankWordDetokenizer()
    return detokenizer.detokenize(tokenized_text)
