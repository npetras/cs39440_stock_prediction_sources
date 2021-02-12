import unittest
import pre_processing as pre_process

class TestPreprocessing(unittest.TestCase):
    def test_punct_basic(self):
        text = """Hello how are you doing today?!"""
        expected_text = "Hello how are you doing today"
        result = pre_process.remove_punct(text)
        self.assertEqual(result, expected_text)

    def test_punct_advanced(self):
        text = """Hello how's are you #twitter doing (fifty) '\n' today?!"""
        expected_text = "Hello hows are you twitter doing fifty n today"
        result = pre_process.remove_punct(text)
        self.assertEqual(result, expected_text)

if __name__=='__main__':
    unittest.main()