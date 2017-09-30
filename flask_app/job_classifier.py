import argparse
import pickle as pickle
import pandas as pd
import numpy as np

from os import path

from string import punctuation, printable

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

from lemmatokenizer import LemmaTokenizer

class JobRecommender(object):
    ''' Create a jobs recommender based on NLP of three text fields:
    1. job description
    2. job title
    3. job majors (e.g. the list of majors required for a specific job)

    ***** NOTE *****
    This is a stupid version to start.  Will become more complex once
    the web app is up and running.
    '''

    def __init__(self):
        self.stop_words = text.ENGLISH_STOP_WORDS.union({'u2019', 'u2020', 'u2022', '\n', '\t', 'u', 'bull', 'nbsp'})
        self._vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words=self.stop_words, ngram_range=(1, 2), max_features=100)
        self._recommender = NMF(n_components=10, alpha=0.1)

    def fit(self, text, n_gram_max=2, num_features=1000):
        ''' Perform TF-IDF vectorization and feed the results into NMF clustering.
        Print the vocabulary words associated with each cluster in order to verify that clustering
        is being performed as-expected.
        '''

        tf = self._vectorizer.fit_transform(text)
        self._recommender.fit_transform(tf)
        return self

    def test_fn_call(self, job):
        '''Test that flask can call the modeling program and use its functionality.
        '''
        return str('This is a test job title')

    def _get_index(self, job_id, jobs_dict):
        ''' Given a job ID, search the jobs_dict dictionary for the
        corresponding index in the cosine similarity matrix and the job
        details table.
        '''
        for key, val in jobs_dict.items():
            if val['jobs_id'] == job_id:
                idx = key
                break
        return key

    def get_job_title(self, jobid, jobs_dict):
        for v in jobs_dict.values():
            if v['jobs_id'] == jobid:
                title = v['jobTitle']
                break
        return title

    def get_full_jobs_list(self, jobs_dict):
        '''Return a sorted list of tuples that contain job titles and job id for
        use in a searchable dropdown list for the app.'''
        titles = [(deets['jobs_id'], str(deets['jobTitle']).strip(punctuation).lstrip())\
                    for deets in jobs_dict.values() if str(deets['jobTitle']).strip(punctuation).lstrip()]
        return sorted(titles, key=lambda x: x[1])

    def make_recommendations(self, job_id, cs_sim_matrix, jobs_dict, k=12):
        '''Sort the cosine similarity matrix from largest to smallest and store
        the results in a list of lists.  The first item will be the job of interest.
        The remaining items will be the top k jobs associated with the job of interest.
        '''
        jobs = cs_sim_matrix[self._get_index(job_id, jobs_dict)]

        top_jobs = jobs.argsort()[:-k - 2:-1]
        ids_list=[jobs_dict[idx]['jobs_id'] for idx in top_jobs]
        jobs_list=[jobs_dict[idx]['jobTitle'] for idx in top_jobs]

        return ids_list, jobs_list

    def get_job_details(self, job_id, jobs_dict):
        '''For a given job title, return the details (description, salary, location and state)
        from the job details table.

        INPUTS:
        job_title (str) - A string containing the job title for which to produce details
        jobs_dict (dict) - A dictionary with index as the key, and value containing
        job details as a dictionary.

        RETURNS:
        The dictionary of job details for a given job (e.g. {'jobDescription': 'Description text',
        'jobLocation': 'Anywhere USA', ...})
        '''
        idx = self._get_index(job_id, jobs_dict)
        if idx in jobs_dict:
            return jobs_dict[idx]
        else:
            return dict()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Perform NLP on career services data and recommend jobs from the output')
    parser.add_argument('--data', help='A pickle file with jobs data.')
    parser.add_argument('--out', help='A file for which to save the pickled model object.')
    args = parser.parse_args()

    df = get_data('data/jobs_with_applicants.pkl')
    # df = get_data(args.data)
    jr = JobRecommender()
    jr.fit(df['jobDescription'])

    picklename = 'static/job_recommender.pkl'
    pickle.dump(jr, open(picklename, 'wb'))

    # with open(args.out, 'wb') as f:
    #     pickle.dump(jr, f)
