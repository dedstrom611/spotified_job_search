import numpy as np
import pandas as pd
from os import path
import re
from string import punctuation, printable

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation as LDA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity

from lemmatokenizer import LemmaTokenizer
#from wordcloud import WordCloud

import matplotlib.pyplot as plt


'''text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
])


parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
}'''

def get_unique_jobid(df, keeplist, byvar):
    ''' Return a dataframe with duplicate job_id's removed and the index reset.
    This will be used to create the cosine similarity matrix for job description.'''
    #jobvars = ['jobs_id', 'jobCompany', 'jobCategory', 'jobDescription', 'jobTitle']
    return df[keeplist].drop_duplicates(byvar).reset_index()

def _lower_strip(s):
    ''' Lowercase and strip punctuation from text before going into TF-IDF
    '''
    s = ''.join([i.lower() for i in s if i not in set(punctuation + '0123456789')])
    return s
    # print(s)

def preprocess_text(df, col):
    # df['new_text_col'] = df[col].apply(lambda x: _lower_strip(x))
    # return df['new_text_col']
    return df[col].apply(lambda x: _lower_strip(x))

def calculate_sparsity(mat):
    matrix_size = mat.shape[0]*mat.shape[1] # Number of possible interactions in the matrix
    num_nonzero = len(mat.nonzero()[0]) # Number of items interacted with
    sparsity = 100*(1 - (num_nonzero/float(matrix_size)))
    return sparsity

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

def make_lookup_dictionary(df, var):
    ''' Create a dictionary of job index and job variable.  This will be used
    later to lookup the job information for the most similar jobs out of the
    similarity matrices.'''
    jobs_dict = {idx: title for idx, title in enumerate(df[var])}
    return jobs_dict

def get_top_jobs(cosine_matrix, jobs_dict, k):
    '''Sort the cosine similarity matrix from largest to smallest and store
    the results in a list of lists.  The first item will be the job of interest.
    The remaining items will be the top k jobs associated with the job of interest.
    '''
    # top_jobs = cs_mat.argsort()[:, :-10:-1]
    top_jobs = cosine_matrix.argsort()[:, :-k - 1:-1]

    jobs_list = []
    for row in top_jobs:
        jobs_k = []
        for idx in row:
            jobs_k.append(jobs_dict[idx])
        jobs_list.append(jobs_k)
    return jobs_list

def softmax(v, temperature=1.0):
    '''
    A heuristic to convert arbitrary positive values into probabilities.
    See: https://en.wikipedia.org/wiki/Softmax_function
    '''
    expv = np.exp(v / temperature)
    s = np.sum(expv)
    return expv / s

def analyze_new_job(W, cluster_index):
    ''' Analyze a new job against a previously computed clustering and
    assign probabilities of belonging to each cluster.
    '''
    W = nmf.fit_transform(X)
    probs = softmax(W[article_index], temperature=0.01)
    for prob, label in zip(probs, hand_labels):
        print ('--> {:.2f}% {}'.format(prob * 100, label))
    print ()

def hand_label_topics(H, vocabulary):
    '''
    Print the most influential words of each latent topic, and prompt the user
    to label each topic. The user should use their humanness to figure out what
    each latent topic is capturing.
    '''
    hand_labels = []
    for i, row in enumerate(H):
        top_ngrams = np.argsort(row)[::-1][:20]
        print ('topic', i)
        print ('-->', ' '.join(vocabulary[top_ngrams]))
        label = raw_input('please label this topic: ')
        hand_labels.append(label)
        print ()
    return hand_labels

def create_topics(clean_text, description=None, n_gram_max=3, num_features=5000, n_topics=10, n_top_words=20):
    ''' Perform TF-IDF vectorization and feed the results into either NMF or LDA clustering.
    Print the vocabulary words associated with each cluster in order to verify that clustering
    is being performed as-expected.
    '''
    tfidf = TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words=stop_words, ngram_range=(1, n_gram_max), max_features=num_features)

    tfidf_model = tfidf.fit_transform(clean_text)
    vocabulary = np.array(tfidf.get_feature_names())

    nmf = NMF(n_components=n_topics, max_iter=max_iterations, random_state=seed, alpha=0.1)
    lda = LDA(n_components=n_topics, learning_method='batch', random_state=seed)

    nmf_model = nmf.fit(tfidf_model)
    lda_model = lda.fit(tfidf_model)

    print("\nTopics in NMF model using TF-IDF Vectorizer: {}".format(description))
    print_top_words(nmf_model, vocabulary, n_top_words)

    print("\nTopics in LDA model using TF-IDF Vectorizer: {}".format(description))
    print_top_words(lda_model, vocabulary, n_top_words)

def fit_nmf_model(clean_text, n_gram_max=3, num_features=5000, n_topics=10, max_iterations=100, seed=1234):
    ''' Perform TF-IDF vectorization and feed the results into either NMF or LDA clustering.
    Print the vocabulary words associated with each cluster in order to verify that clustering
    is being performed as-expected.
    '''
    stop_words = text.ENGLISH_STOP_WORDS.union({'u2019', 'u2020', 'u2022', '\n', '\t', 'u', 'bull', 'nbsp'})

    tfidf = TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words=stop_words, ngram_range=(1, n_gram_max), max_features=num_features)

    tfidf_model = tfidf.fit_transform(clean_text)
    vocabulary = np.array(tfidf.get_feature_names())

    nmf = NMF(n_components=n_topics, max_iter=max_iterations, random_state=seed, alpha=0.1)
    W = nmf.fit_transform(tfidf_model)
    H = nmf.components_

    return vocabulary, W, H

def _pull_all_caps(text):
    s = ''.join([i for i in text if i not in set(punctuation + '0123456789')])
    keep_all_caps = ' '.join(word.lower() + word[1:] if not word.isupper() else word for word in s.split())
    return re.sub('[^A-Z]', '', keep_all_caps)

def pull_all_caps(df, col):
    return df[col].apply(lambda row: _pull_all_caps(row))
    # Keep words in all caps.

    s = ''.join([i.lower() for i in s if i not in set(punctuation + '0123456789')])
    keep_all_caps = ' '.join(word.lower() + word[1:] if not word.isupper() else word for word in string.split())

def get_stop_words():
    from sklearn.feature_extraction import text
    return text.ENGLISH_STOP_WORDS.union({'u2019', 'u2020', 'u2022', '\n', '\t', 'u', 'bull', 'nbsp'})

# def print_word_cloud(text):
#     # lower max_font_size
#     wordcloud = WordCloud(max_font_size=40).generate(text)
#     plt.figure()
#     plt.imshow(wordcloud, interpolation="bilinear")
#     plt.axis("off")

if __name__ == '__main__':
    seed = 1234
    n_topics = 10
    n_top_words = 25
    max_iterations = 200

    df = pd.read_pickle('../data/posted_jobs.pkl')
    # appsdf = pd.read_pickle('../data/jobs_with_applicants.pkl')

    df = df.query("applicationCount == 0")
    stop_words = text.ENGLISH_STOP_WORDS.union({'u2019', 'u2020', 'u2022', '\n', '\t', 'u', 'bull', 'nbsp'})

    # create_topics(clean_descs)
    # create_topics(clean_titles, n_gram_max=2, num_features=500, n_topics=10, n_top_words=20)
    # create_topics(appsdf['major_text'], description = 'Job Majors', n_gram_max=2, num_features=500, n_topics=10, n_top_words=15)

    vocab, W_d, H_d = fit_nmf_model(df['desc_text'], n_gram_max=3, num_features=5000, n_topics=10)
    vocab, W_m, H_m = fit_nmf_model(df['major_text'], n_gram_max=2, num_features=500, n_topics=10)
    vocab, W_t, H_t = fit_nmf_model(df['title_text'], n_gram_max=2, num_features=500, n_topics=10)

    # ''' Fit the TF model to the text data '''
    # tf = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words=stop_words, ngram_range=(1, 3), max_features=5000)
