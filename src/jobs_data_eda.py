import pandas as pd
import numpy as np
from string import punctuation

def print_missing_values(df):
    '''Print the pct. of missing values and return a dictionary of the columns and
    pct missing for each'''
    n = df.shape[0]

    print('Column Name \t Pct. Missing \n')
    print('-------------------------------')
    for col in df.columns:
        pct_miss = (df[col].isnull().values.ravel().sum() / float(n))*100
        print('{}:\t{:.2f}'.format(col, pct_miss))

def query_dataframe(df, col, type_description):
    '''Keep only rows of the jobs dataframe where jobType is in a string

    INPUTS:
    df - Pandas dataframe
    col - Column in the pandas dataframe to query
    type_description - list of str's, description to keep (e.g. ['Full Time'])

    RETURNS:
    Pandas dataframe subset with the rows that meet type_description criteria
    '''
    return df.loc[df[col].isin(type_description)]

def _lower_strip(s):
    ''' Lowercase and strip punctuation from text before going into TF-IDF
    '''
    s = ' '.join(i for i in s.split(','))
    s2 = ''.join(i.lower() for i in s if i not in set(punctuation + '0123456789'))
    return s2

def preprocess_text(df, col):
    # df['new_text_col'] = df[col].apply(lambda x: _lower_strip(x))
    # return df['new_text_col']
    return df[col].apply(lambda x: _lower_strip(x))

def remove_bad_words(text):
    remove_list = ['&nbsp', '&bull', '%u2019', '%u2020',\
                   '%u2022', '\n', '\t']
    for term in remove_list:
        text = text.replace(term, '')
    return term

if __name__ == '__main__':
    posted_jobs = pd.read_json('../data/jobs_posted.json')
    applicants = pd.read_json('../data/job_applications.json')
    student_info = pd.read_json('../data/student_info.json')
    student_aboutme = pd.read_json('../data/student_aboutme.json')

    # fix_cols = ['jobDescription', 'jobMajor', 'jobTitle']
    # for col in fix_cols:
    #     posted_jobs[col] = posted_jobs[col].apply(lambda x: remove_bad_words(x))
    #
    # student_aboutme['aboutMe'] = student_aboutme['aboutMe'].apply(lambda x: remove_bad_words(x))

    # Pre-process the text fields of interest for later use in text analysis.
    posted_jobs['desc_text'] = preprocess_text(posted_jobs, 'jobDescription')
    posted_jobs['major_text'] = preprocess_text(posted_jobs, 'jobMajor')
    posted_jobs['title_text'] = preprocess_text(posted_jobs, 'jobTitle')
    student_aboutme['aboutme_text'] = preprocess_text(student_aboutme, 'aboutMe')

    # Merge student information together
    tmp1 = pd.merge(applicants, student_info, how='left', on='accounts_id')
    student_df = pd.merge(tmp1, student_aboutme, how='left', on='accounts_id')

    # Subset only jobs that have applicants
    jobs_wapps = posted_jobs.query("applicationCount > 0")

    # merge jobs/apps data with student preferences data
    df = pd.merge(jobs_wapps, student_df, how='left', on='jobs_id')

    # Create pickled versions of the dataframes.
    df.to_pickle('../data/jobs_apps_dataset.pkl')
    jobs_wapps.to_pickle('../data/jobs_with_applicants.pkl')
    posted_jobs.to_pickle('../data/posted_jobs.pkl')

    # Write the sparse results to a JSON file for use in Spark
    # df.to_json('../data/job_apps_prefs_final.json')
    # sparse.to_json('../data/job_app_sparse.json')
