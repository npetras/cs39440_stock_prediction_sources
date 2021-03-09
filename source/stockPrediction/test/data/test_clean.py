"""
Test module for the module main.data.clean
"""
import unittest
# pylint: disable=E0611
# issue with nltk, or bad error from pylint
import nltk.corpus.reader.wordnet as wordnet_corpus
from main.data import clean

GEORGIA_HEADLINE_LC = 'georgia downs two russian warplanes as countries move to brink of war'
GEORGIA_HEADLINE_STEMMED = 'georgia down two russian warplan as countri move to brink of war'
GEORGIA_HEADLINE_LEMMATIZED = "georgia down two russian warplane as country move to brink of war"
CAT_TEXT_LIST = list('The cat is sitting in the window looking out at the street'.split(' '))


class TestClean(unittest.TestCase):
    """
    Test class for the module main.data.clean
    """

    def test_sklearn_tokenize_basic(self):
        """
        Provides a basic string to sklearn_tokenize()
        """
        text = '''hello how are you?!'''
        expected_list = ['hello', 'how', 'are', 'you']
        resulting_list = clean.sklearn_tokenize(text)
        self.assertEqual(expected_list, resulting_list)

    def test_sklearn_tokenize_advanced(self):
        """
        Provides a more advanced string with enclitics, brackets in the text, and a singular
        one-letter word (i)
        """
        text = '''Hello i didn't see you (at the talk), were you there?'''
        expected_list = ['Hello', 'didn', 'see', 'you', 'at', 'the', 'talk', 'were', 'you', 'there']
        resulting_list = clean.sklearn_tokenize(text)
        self.assertEqual(expected_list, resulting_list)

    def test_treebank_detokenize(self):
        """
        Provides a tokenized list, with no punctuation as is expected after the sklearn tokenizer is
        run on the text
        """
        tokenized_text = ['Hello', 'didn', 'see', 'you', 'at', 'the', 'talk', 'were', 'you',
                          'there']
        expected_text = 'Hello didn see you at the talk were you there'
        resulting_text = clean.treebank_detokenize(tokenized_text)
        self.assertEqual(expected_text, resulting_text)

    def test_porter_stem_basic(self):
        """
        Basic string provided to porter_stem()
        """
        text = CAT_TEXT_LIST
        expected_text = list('the cat is sit in the window look out at the street'.split(' '))
        result = clean.porter_stem(text)
        self.assertEqual(expected_text, result)

    def test_porter_stem_headline(self):
        """
        Headline string from DJIA dataset provided to porter_stem()
        """
        text = list(GEORGIA_HEADLINE_LC.split(' '))
        expected_text = list(GEORGIA_HEADLINE_STEMMED.split(' '))
        result = clean.porter_stem(text)
        self.assertEqual(expected_text, result)

    def test_porter_stem_list_headlines(self):
        """
        Provides a list of untokenized headlines
        """
        headline1 = GEORGIA_HEADLINE_LC
        headline2 = 'BREAKING Musharraf to be impeached'
        headline3 = 'Russia Today Columns of troops roll into South Ossetia footage from ' \
                    'fighting YouTube'
        text_list = [headline1, headline2, headline3]
        expected_headline1 = GEORGIA_HEADLINE_STEMMED
        expected_headline2 = 'break musharraf to be impeach'
        expected_headline3 = 'Russia Today Column of troop roll into South Ossetia footag ' \
                             'from fight YouTub'.lower()
        expected_list = [expected_headline1, expected_headline2, expected_headline3]
        resulting_list = clean.porter_stem_list(text_list)
        self.assertEqual(expected_list, resulting_list)

    def test_simply_pos_tag_proper_noun(self):
        """
        Provides simple_pos_tag() the proper noun Penn Treebank part of speech (POS) tag.
        """
        proper_noun_tag = 'NNP'
        expected_tag = wordnet_corpus.NOUN
        result = clean.simple_pos_tag(proper_noun_tag)
        self.assertEqual(expected_tag, result)

    def test_simply_pos_tag_verb(self):
        """
        Provides simple_pos_tag() the verb Penn Treebank part of speech (POS) tag.
        """
        verb_tag = 'VB'
        expected_tag = wordnet_corpus.VERB
        result = clean.simple_pos_tag(verb_tag)
        self.assertEqual(expected_tag, result)

    def test_simply_pos_tag_comparitive_adjective(self):
        """
        Provides simple_pos_tag() the comparative adjective Penn Treebank part of speech (POS) tag.
        """
        proper_noun_tag = 'JJR'
        expected_tag = wordnet_corpus.ADJ
        result = clean.simple_pos_tag(proper_noun_tag)
        self.assertEqual(expected_tag, result)

    def test_simply_pos_tag_proper_adverb(self):
        """
        Provides simple_pos_tag() the adverb Penn Treebank part of speech (POS) tag.
        """
        proper_noun_tag = 'RB'
        expected_tag = wordnet_corpus.ADV
        result = clean.simple_pos_tag(proper_noun_tag)
        self.assertEqual(expected_tag, result)

    def test_simply_pos_tag_proper_determiner(self):
        """
        Provides simple_pos_tag() the determiner Penn Treebank part of speech (POS) tag.
        """
        proper_noun_tag = 'DT'
        expected_tag = None
        result = clean.simple_pos_tag(proper_noun_tag)
        self.assertEqual(expected_tag, result)

    def test_wordnet_lemmatize_basic(self):
        """
        Provides a basic sentence to be lemmatized
        """
        text = CAT_TEXT_LIST
        expected_text = list('The cat be sit in the window look out at the street'.split(' '))
        result = clean.wordnet_lemmatize(text)
        self.assertEqual(result, expected_text)

    def test_wordnet_lemmatize_simple_headline(self):
        """
        Provides a headline to be lemmatized
        """
        text = list(GEORGIA_HEADLINE_LC.split(' '))
        expected_text = list(GEORGIA_HEADLINE_LEMMATIZED.split(' '))
        result = clean.wordnet_lemmatize(text)
        self.assertEqual(result, expected_text)

    def test_word_lemmatize_list(self):
        """
        Provides a list of untokenized headlines
        """
        headline1 = GEORGIA_HEADLINE_LC
        headline2 = 'BREAKING Musharraf to be impeached'
        headline3 = 'Russia Today Columns of troops roll into South Ossetia footage from ' \
                    'fighting YouTube'
        text_list = [headline1, headline2, headline3]
        expected_headline1 = GEORGIA_HEADLINE_LEMMATIZED
        expected_headline2 = 'BREAKING Musharraf to be impeach'
        expected_headline3 = 'Russia Today Columns of troop roll into South Ossetia footage ' \
                             'from fight YouTube'
        expected_list = [expected_headline1, expected_headline2, expected_headline3]
        resulting_list = clean.wordnet_lemmatize_list(text_list)
        self.assertEqual(expected_list, resulting_list)


if __name__ == '__main__':
    unittest.main()
