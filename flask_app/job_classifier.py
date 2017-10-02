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
    ''' Use an existing cosine similarity matrix and a dictionary of job details
    in order to fetch recommended jobs/job details.  The results interact with a
    flask app in order to allow searching of existing jobs and the return of a set
    of related, recommended jobs.

    ARGUMENTS:
    None
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

    def _jobs_in_state_list(self, jobs_list, state_list, jobs_dict):
        ''' Given a list of job indices, check each job and return a new list containing
        only jobs that are in the state list.  NOTE: Also return a job if the state is
        any of ['national', 'Nationwide', 'None', or '--- United States of America ---'].

        If the state list is empty, then return the original jobs list back.

        INPUTS:
        jobs_list (list) - A list of indices for recommended jobs.  This is the original
        list that will have non-relevant states removed.

        state_list (list) - A list containing all state abbreviations that the user has
        selected from the web app e.g. ['GA', 'TX'].  If no states are selected, then
        state_list = []

        jobs_dict (dictionary) - A dictionary of dictionaries where key is the job index
        and a dictionary of job details are the values
        e.g. {3423: {'jobs_id'}: 9023413, {'jobTitle'}: 'Sales Manager', ... }

        RETURNS:
        A list of the job indices that meet the state criteria.
        '''
        if not state_list:
            return jobs_list
        else:
            id_list = []
            # If job is in a state that is not state specific...
            for idx in jobs_list:
                if (not jobs_dict[idx]['jobState']) or \
                (jobs_dict[idx]['jobState'] == 'national') or \
                (jobs_dict[idx]['jobState'] == '--- United States of America ---') or \
                (jobs_dict[idx]['jobState'] == 'Nationwide'):
                    id_list.append(idx)
                else:
                    # If the job has a state or list of states associated with it...
                    for state in state_list:
                        if state in jobs_dict[idx]['jobState']:
                            id_list.append(idx)
                            break
            return id_list

    def make_recommendations(self, job_id, cs_sim_matrix, jobs_dict, state_list, k=12):
        '''Sort the cosine similarity matrix from largest to smallest and store
        the results in a list of lists.  The first item will be the job of interest.
        The remaining items will be the top k jobs associated with the job of interest.
        If the user has selected specific states, perform initial filtering to choose
        only jobs in those states.
        '''
        jobidx = cs_sim_matrix[self._get_index(job_id, jobs_dict)].argsort()[len(cs_sim_matrix)::-1]

        top_jobs = np.append(jobidx[0], self._jobs_in_state_list(jobidx[1:], state_list, jobs_dict))

        #top_jobs = jobs.argsort()[:-k - 2:-1]
        ids_list=[jobs_dict[idx]['jobs_id'] for idx in top_jobs]
        jobs_list=[jobs_dict[idx]['jobTitle'] for idx in top_jobs]

        return ids_list[0: k], jobs_list[0: k]

    def get_job_details(self, job_id, jobs_dict):
        '''For a given job ID, return the details (description, salary, location and state)
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
