import unittest
import preprocess

class TestPreprocessing(unittest.TestCase):
    def test_to_lowercase_1(self):
        text = "HELLO HOW ARE YOU?"
        expected_text = "hello how are you?"
        result = preprocess.to_lowercase(text)
        self.assertEqual(result, expected_text)
    
    def test_to_lowecase_2(self):
        text = "HELLO how are YOU?"
        expected_text = "hello how are you?"
        result = preprocess.to_lowercase(text)
        self.assertEqual(result, expected_text)
    
    def test_punct_basic_string(self):
        text = """Hello how are you doing today?!"""
        expected_text = "Hello how are you doing today"
        result = preprocess.remove_punct(text)
        self.assertEqual(result, expected_text)

    def test_punct_complex_string(self):
        text = """Hello how's didn't are you #twitter doing (fifty) '{n}' today?!"""
        expected_text = "Hello how s didn t are you twitter doing fifty n today"
        result = preprocess.remove_punct(text)
        self.assertEqual(result, expected_text)
    
    def test_remove_stopwords_basic_stopwords(self):
        text  = "georgia 'downs two russian warplanes' as countries move to brink of war"
        expected_text = "georgia 'downs two russian warplanes' countries move brink war"
        result = preprocess.remove_stopwords(text)
        self.assertEqual(result, expected_text)

    def test_remove_stopwords_basic_stopwords2(self):
        text  = """I Know We've Got This Election in US To Worry, But I Wanted To Help Shine More Light On Our Australian Friends Potential 'Internet Filter' Problem. Let's Keep That In The Spotlight Too!""".lower()
        expected_text = "Know've Got Election US Worry, Wanted Help Shine Light Australian Friends Potential 'Internet Filter' Problem . Let's Keep Spotlight!".lower()
        result = preprocess.remove_stopwords(text)
        self.assertEqual(result, expected_text)

    def test_remove_stopwords_no_punct_string(self):
        text  = """I Know We ve Got This Election in US To Worry But I Wanted To Help Shine More Light On Our Australian Friends Potential Internet Filter Problem Let s Keep That In The Spotlight Too""".lower()
        expected_text = """Know Got Election US Worry Wanted Help Shine Light Australian Friends Potential Internet Filter Problem Let Keep Spotlight""".lower()
        result = preprocess.remove_stopwords(text)
        self.assertEqual(result, expected_text)

    def test_apply_all_1(self):
        text  = """I Know We've Got This Election in US To Worry, But I Wanted To Help Shine More Light On Our Australian Friends Potential 'Internet Filter' Problem. Let's Keep That In The Spotlight Too!"""
        expected_text  = """Know Got Election US Worry Wanted Help Shine Light Australian Friend Potential Internet Filter Problem Let Keep Spotlight""".lower()
        # result = preprocess.apply_all(text)
        # self.assertEqual(result, expected_text)

    def test_apply_all_2(self):
        text  = """ """
        expected_text  = """ """.lower()
        # result = preprocess.apply_all(text)
        # self.assertEqual(result, expected_text)
if __name__=='__main__':
    unittest.main()