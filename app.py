import pickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
# from typing import List

model = None
app = Flask(__name__)
# cors
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# department = 'Sales'
# job_role = 'Manager'
# job_level = 2


# model variable refers to the global variable
with open('lgb_model', 'rb') as f:
    model = pickle.load(f)

train_cols = pickle.load(open('train_column', 'rb')).columns
linkedin_data = pd.read_csv('result_gender.csv').drop('Name', axis=1)
categorical = ['Gender', 'EducationField', 'Department', 'JobRole']
numerical = ['Education', 'Age', 'TotalWorkingYears','NumCompaniesWorked', 'JobLevel']

def scrape_analysis(department, job_role, job_level):
    predictions = []
    recommend_index = []
    recommend = []

    scrape_data = linkedin_data.assign(Department=pd.Series(department, index=linkedin_data.index))
    scrape_data = scrape_data.assign(JobRole=pd.Series(job_role, index=scrape_data.index))
    scrape_data = scrape_data.assign(JobLevel=pd.Series(job_level, index=scrape_data.index))
    
    df_num = scrape_data[numerical]
    df_cat = scrape_data[categorical]
    df_cat = pd.get_dummies(df_cat)
    df_scrape_missing = pd.concat([df_num, df_cat], axis=1)

    # deal with missing columns
    missing_cols = set(train_cols) - set(df_scrape_missing.columns)
    for col in missing_cols:
        df_scrape_missing[col] = 0
    df_scrape = df_scrape_missing[train_cols]

    # predict
    result = model.predict(df_scrape)
    
    # result
    for x in result:
        max = x.argsort()[-1] 
        if x[max]>0.8:
            predictions.append(max)
        elif max == 0:
            predictions.append(max)
        else:
            predictions.append(max-1)
    
    for index, pred in enumerate(predictions):
        if pred == 2:
            recommend_index.append(index)

    for i in recommend_index:
        recommend.append(scrape_data.iloc[i].URL)
    return recommend


def analysis(gender, education, education_field, age, working_year, company_num, department, job_role, job_level):
    predictions = []
    recommend_index = []
    recommend = []
    column = ['Gender', 'Education', 'EducationField', 'Age', 'TotalWorkingYears', 'NumCompaniesWorked', 'Department', 'JobRole', 'JobLevel']
    data = pd.DataFrame([[gender, education, education_field, age, working_year, company_num, department, job_role, job_level]], columns=column)
    
    df_num = data[numerical]
    df_cat = data[categorical]
    df_cat = pd.get_dummies(df_cat)
    df_missing = pd.concat([df_num, df_cat], axis=1)

    # deal with missing columns
    missing_cols = set(train_cols) - set(df_missing.columns)
    for col in missing_cols:
        df_missing[col] = 0
    df = df_missing[train_cols]

    # predict
    result = model.predict(df)
    print(result)
    # result
    for x in result:
        max = x.argsort()[-1] 
        if x[max]>0.8:
            predictions.append(max)
        elif max == 0:
            predictions.append(max)
        else:
            predictions.append(max-1)
    return str(predictions[0])


@app.route('/')
@cross_origin()
def home():
    return '200 OK'


@app.route('/scraping', methods=['GET'])
@cross_origin()
def get_scraping():
    # if request.method == 'GET':
    # data = request.get_json()  # Get data posted as a json
    # data = np.array(data)[np.newaxis, :]  # converts shape from (4,) to (1, 4)
    # prediction = model.predict(data)  # runs globally loaded model on the data
    department = request.args.get('department')
    job_role = request.args.get('job_role')
    job_level = int(request.args.get('job_level'))
    print(department, job_role, job_level)
    result = scrape_analysis(department, job_role, job_level)
    return jsonify(total=linkedin_data.shape[0], recommendation=result) # number of row count


@app.route('/predict', methods=['GET'])
@cross_origin()
def get_prediction():
    gender = request.args.get('gender')
    education = int(request.args.get('education'))
    education_field = request.args.get('education_field')
    age = int(request.args.get('age'))
    working_year = int(request.args.get('working_year'))
    company_num = int(request.args.get('company_num'))
    department = request.args.get('department')
    job_role = request.args.get('job_role')
    job_level = int(request.args.get('job_level'))
    result = analysis(gender, education, education_field, age, working_year, company_num, department, job_role, job_level)
    return result


if __name__ == '__main__':
    # load_model()  # load model at the beginning once only
    app.run(debug=True)
