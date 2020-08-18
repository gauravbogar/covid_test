from flask import jsonify, request, Flask
import pandas as pd
import numpy as np 

app = Flask(__name__)




@app.route('/api/covid_test')
def predict_risk():
    data = request.get_json()
    sex = data['sex']
    pneumonia = data['pneumonia']
    age = data['age']
    pregnancy = data['pregnancy']
    diabetes = data['diabetes']
    copd = data['copd']
    asthma = data['asthma']
    hypertension = data['hypertension']
    other_disease = data['other_disease']
    cardiovascular = data['cardiovascular']
    obesity = data['obesity']
    renal_chronic = data['renal_chronic']
    tobacco = data['tobacco']
    contact_other_covid = data['contact_other_covid']
    renal_chronic = data['renal_chronic']

    cols = ['sex', 'pneumonia', 'age', 'pregnancy', 'diabetes', 'copd', 'asthma',
       'hypertension', 'other_disease', 'cardiovascular', 'obesity',
       'renal_chronic', 'tobacco', 'contact_other_covid', 'covid_res']
    
    test = pd.Dataframe([[sex, pneumonia, age, pregnancy, diabetes, copd, asthma,
       hypertension, other_disease, cardiovascular, obesity,
       renal_chronic, tobacco, contact_other_covid,]], columns = cols)
    
    pred = classifier.predict_proba(test)