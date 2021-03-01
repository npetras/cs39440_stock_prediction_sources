import clean_data
import unittest

US_ELECTION_HEADLINE_RAW = """I Know We've Got This Election in US To Worry, But I Wanted To Help Shine More Light On Our Australian Friends Potential 'Internet Filter' Problem. Let's Keep That In The Spotlight Too!"""
US_ELECTION_HEADLINE_RAW_LC = US_ELECTION_HEADLINE_RAW.lower()

class TestPreprocessing(unittest.TestCase):

    def test_to_lowercase_all_caps(self):
        text = "HELLO HOW ARE YOU?"
        expected_text = "hello how are you?"
        result = clean_data.to_lowercase(text)
        self.assertEqual(result, expected_text)
    
    def test_to_lowecase_mixed_caps(self):
        text = "HELLO how are YOU?"
        expected_text = "hello how are you?"
        result = clean_data.to_lowercase(text)
        self.assertEqual(result, expected_text)
    
    def test_to_lowercase_headline(self):
        text = US_ELECTION_HEADLINE_RAW
        expected_text = US_ELECTION_HEADLINE_RAW_LC
        result = clean_data.to_lowercase(text)
        self.assertEqual(result, expected_text)
    
    def test_sklearn_tokenize_basic(self):
        text = """Hello how are you doing today?!"""
        expected_list = ['Hello', 'how', 'are', 'you', 'doing', 'today']
        result = clean_data.sklearn_tokenize(text)
        self.assertEqual(result, expected_list)

    def test_sklearn_tokenize_complex(self):
        text = """Hello how's didn't are you #twitter doing (fifty) '{n}' today?!"""
        expected_list = ['Hello', "how", "didn", 'are', 'you', 'twitter', 'doing', 'fifty', 'today']
        result = clean_data.sklearn_tokenize(text)
        self.assertEqual(result, expected_list)
    
    def test_sklearn_tokenize_headline(self):
        text  = "b'Rice Gives Green Light for Israel to Attack Iran: Says U.S. has no veto over Israeli military ops'"
        expected_list = ['Rice', 'Gives', 'Green', 'Light', 'for', 'Israel', 'to', 'Attack', 'Iran', 'Says', 'has', 'no', 'veto', 'over', 'Israeli', 'military', 'ops']
        result = clean_data.sklearn_tokenize(text)
        self.assertEqual(result, expected_list)

    def test_detokenize_basic(self):
        tokenized_list = ['Hello', "how", "didn", 'are', 'you', 'twitter', 'doing', 'fifty', 'today']
        detokenized_text = 'Hello how didn are you twitter doing fifty today'
        result = clean_data.detokenize(tokenized_list)
        self.assertEqual(result, detokenized_text)

    def test_detokenize_headline(self):
        tokenized_list = ['Rice', 'Gives', 'Green', 'Light', 'for', 'Israel', 'to', 'Attack', 'Iran', 'Says', 'has', 'no', 'veto', 'over', 'Israeli', 'military', 'ops']
        detokenized_text = 'Rice Gives Green Light for Israel to Attack Iran Says has no veto over Israeli military ops'
        result = clean_data.detokenize(tokenized_list)
        self.assertEqual(result, detokenized_text)


    
    def test_remove_stopwords_basic(self):
        tokenized_list = ['hello', "how", "didn", 'are', 'you', 'twitter', 'doing', 'fifty', 'today']
        expected_list = ['hello', 'twitter', 'fifty', 'today']
        result = clean_data.remove_stopwords(tokenized_list)
        self.assertEqual(result, expected_list)

    def test_remove_stopwords_short_headline(self):
        text  = list("georgia downs two russian warplanes as countries move to brink of war".split(' '))
        expected_text = list("georgia downs two russian warplanes countries move brink war".split(' '))
        result = clean_data.remove_stopwords(text)
        self.assertEqual(result, expected_text)

    def test_remove_stopwords_long_headline_no_punct(self):
        text  = list("""I Know We ve Got This Election in US To Worry But I Wanted To Help Shine More Light On Our Australian Friends Potential Internet Filter Problem Let s Keep That In The Spotlight Too""".lower().split(' '))
        expected_text = list("""Know Got Election US Worry Wanted Help Shine Light Australian Friends Potential Internet Filter Problem Let Keep Spotlight""".lower().split(' '))
        result = clean_data.remove_stopwords(text)
        self.assertEqual(result, expected_text)

    def test_wordnet_lemmatize_basic1(self):
        text = list('The cat is sitting in the window looking out at the street'.split(' '))
        expected_text = list("The cat be sit in the window look out at the street".split())
        result = clean_data.word_net_lemmatize(text)
        self.assertEqual(result, expected_text)

    def test_wordnet_lemmatize_simple_headline(self):
        text  = list("georgia downs two russian warplanes as countries move to brink of war".split(' '))
        expected_text = list("georgia down two russian warplane as country move to brink of war".split(' '))
        result = clean_data.word_net_lemmatize(text)
        self.assertEqual(result, expected_text)
    
    def test_porter_stem_basic(self):
        text = list('The cat is sitting in the window looking out at the street'.split(' '))
        expected_text = list("the cat is sit in the window look out at the street".split())
        result = clean_data.porter_stem(text)
        self.assertEqual(result, expected_text)

    def test_porter_stem_basic(self):
        text  = list("georgia downs two russian warplanes as countries move to brink of war".split(' '))
        expected_text = list("georgia down two russian warplan as countri move to brink of war".split(' '))
        result = clean_data.porter_stem(text)
        self.assertEqual(result, expected_text)

if __name__=='__main__':
    unittest.main()
