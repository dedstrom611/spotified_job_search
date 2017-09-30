from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

from job_classifier import JobRecommender

app = Flask(__name__)

# with open('static/job_recommender.pkl', 'rb') as f:
#     model = pickle.load(f)

model = JobRecommender()

with open('../data/jobs_similarity_matrix.pkl', 'rb') as m:
    matrix = pickle.load(m)

with open('../data/job_details_dict.pkl', 'rb') as det:
    details_dict = pickle.load(det)

@app.route('/', methods=['POST', 'GET'])
def index():
    """Render a simple splash page."""
    return render_template('cover.html')

@app.route('/submit', methods=['POST', 'GET'])
def submit():
    """Render a page containing a text input where the user can enter a
    job for which to find recommendations.  """
    jobs = list(model.get_full_jobs_list(details_dict))
    return render_template('submit.html', full_job_list=jobs)

# def search_job():
#     all_jobs = map(lambda x: x.upper(), df['jobTitle'])
#     results = [job if search_title.upper() in job for job in results]

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Recieve the job from an input form and use the
    model to classify.
    """
    jobid = int(request.form['job_title'])
    jobtitle = str(model.get_job_title(jobid, details_dict))
    ids, titles = model.make_recommendations(jobid, matrix, details_dict)

    # Get the list of job details for each recommended job.
    details_list = [model.get_job_details(i, details_dict) for i in ids]
    return render_template('predict_psuedo.html', job=jobtitle, details=details_list)

# @app.route('/results', methods=['GET', 'POST'])
# def results():
#     """Recieve the article to be classified from an input form and use the
#     model to classify.
#     """
#     data = str(request.form['job_title'])
#     details = dict(model.get_job_details(data, details_dict))
#     return render_template('results.html', job=data, jobinfo=details)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
