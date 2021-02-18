"""
https://www.nltk.org/book/ch01.html
"""
def lexical_diversity(text):
    return len(set(text)) / len(text)

# different to nltk book version
def calculate_percentage(a, b):
    return 100 * a / b