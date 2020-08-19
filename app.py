import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

current_path = os.getcwd()
pickle_path = os.path.join(current_path, "assets", "covid.pkl")
print(pickle_path)
classifier = pickle.load(open(pickle_path, "rb"))


@app.route("/api/covid_test", methods=["POST"])
def predict_risk():
    data = request.get_json()
    sex = data["sex"]
    pneumonia = data["pneumonia"]
    age = data["age"]
    pregnancy = data["pregnancy"]
    diabetes = data["diabetes"]
    copd = data["copd"]
    asthma = data["asthma"]
    hypertension = data["hypertension"]
    other_disease = data["other_disease"]
    cardiovascular = data["cardiovascular"]
    obesity = data["obesity"]
    renal_chronic = data["renal_chronic"]
    tobacco = data["tobacco"]
    contact_other_covid = data["contact_other_covid"]

    cols = [
        "sex",
        "pneumonia",
        "age",
        "pregnancy",
        "diabetes",
        "copd",
        "asthma",
        "hypertension",
        "other_disease",
        "cardiovascular",
        "obesity",
        "renal_chronic",
        "tobacco",
        "contact_other_covid",
    ]

    test = pd.DataFrame(
        [
            [
                sex,
                pneumonia,
                age,
                pregnancy,
                diabetes,
                copd,
                asthma,
                hypertension,
                other_disease,
                cardiovascular,
                obesity,
                renal_chronic,
                tobacco,
                contact_other_covid,
            ]
        ],
        columns=cols,
    )

    pred = classifier.predict_proba(test)
    result = pred[0][1]

    if result < 0.25:
        return jsonify("Low Risk")
    elif (result >= 0.25) and (result < 0.5):
        return jsonify("Medium Risk")

    elif (result >= 0.5) and (result < 0.75):
        return jsonify("High Risk")

    else:
        return jsonify("Vulnerable Risk")


if __name__ == "__main__":
    app.run(debug=True)
