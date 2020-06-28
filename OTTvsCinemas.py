from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
from werkzeug import secure_filename
import os
import numpy as np
import joblib as joblib

loaded_model=joblib.load("model.pkl")
loaded_vec=joblib.load("vectorizer.pkl")

app = Flask(__name__)

def classify(document):
    label = {0: 'Neutral', 1: 'Cinemas', 2 : 'OTT'}
    X = loaded_vec.transform([document])
    X = X.toarray()
    y = loaded_model.predict(X)[0]
    proba = np.max(loaded_model.predict_proba(X))
    return label[y], proba

class ReviewForm(Form):
    moviereview = TextAreaField('',[validators.DataRequired(),validators.length(min=15)])

@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('OTTvsCinemas_Input.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['moviereview']
        y, proba = classify(review)
        return render_template('OTTvsCinemas_Results.html',content=review,prediction=y,probability=round(proba*100, 2))
        return render_template('OTTvsCinemas_Input.html', form=form)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)