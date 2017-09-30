from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

if __name__ == "__main__":
	tfidf = TfidfVectorizer(tokenizer=LemmaTokenizer(),
							stop_words='english')
