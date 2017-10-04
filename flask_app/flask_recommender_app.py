from flask import Flask, render_template, request, json, jsonify
import pickle
import pandas as pd
import numpy as np

import plotly
import plotly.plotly as py
import plotly.graph_objs as go
#The line below will let you run this offline.
from plotly.offline import download_plotlyjs,plot, iplot
import pickle

from job_classifier import JobRecommender

app = Flask(__name__)

# with open('static/job_recommender.pkl', 'rb') as f:
#     model = pickle.load(f)

model = JobRecommender()

with open('../data/jobs_similarity_matrix2.pkl', 'rb') as m:
    matrix = pickle.load(m)

with open('../data/job_details_dict.pkl', 'rb') as det:
    details_dict = pickle.load(det)

with open('../data/descrip_nmf_matrix.pkl', 'rb') as f:
    W = pickle.load(f)

@app.route('/', methods=['POST', 'GET'])
def index():
    """Render a simple splash page."""
    return render_template('cover.html')

@app.route('/about', methods=['POST', 'GET'])
def about():
    """Render the about page if clicked. """
    return render_template('about.html')

@app.route('/contact', methods=['POST', 'GET'])
def contact():
    """Render the contact page if clicked. """
    return render_template('contact.html')

@app.route('/submit', methods=['POST', 'GET'])
def submit():
    """Render a page containing a text input where the user can enter a
    job for which to find recommendations.  """
    jobs = list(model.get_full_jobs_list(details_dict))
    return render_template('submit.html', full_job_list=jobs)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Recieve the list of states and the job from an input form and return
    job recommendations.
    """
    states = request.form.getlist('states')
    jobid = int(request.form['job_title'])
    jobtitle = str(model.get_job_title(jobid, details_dict))
    ids, titles = model.make_recommendations(jobid, matrix, details_dict, states)

    # Get the list of job details for each recommended job.
    details_list = [model.get_job_details(i, details_dict) for i in ids]

    return render_template('predict.html', job=jobtitle, details=details_list, states=states, jobids = ids)

@app.route('/get_jobs_in_selected_states', methods=['POST','GET'])
def get_jobs_in_selected_states():
    states = request.args.get('states')
    data = model.get_subset_jobs_list(states, details_dict)
    new_dict = {"id": data[0], "title": data[1]}
    return render_template('get_jobs_in_selected_states.html', test=new_dict, states=states)
    # return jsonify(new_dict)

@app.route('/3dplot_template', methods=['POST','GET'])
def plot_template():
    x = list(W[:,1].flatten())
    y = list(W[:,9].flatten())
    z = list(W[:,6].flatten())
    c = list(np.argmax(W, axis=1))

    return render_template('3dplot_template.html', x=json.dumps(x), y=json.dumps(y), z=json.dumps(z), c=c)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
