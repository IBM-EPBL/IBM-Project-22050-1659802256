# Importing essential libraries
from flask import Flask, render_template,send_file, request
import pickle
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the Random Forest CLassifier model
filename = 'heart-disease-prediction-knn-model.pkl'
model = pickle.load(open(filename, 'rb'))
global age 
global sex 
global cp
global trestbps
global chol
global fbs
global restecg
global thalach
global exang
global oldpeak
global slope
global ca
global thal

app = Flask(__name__,static_url_path='/static')

@app.route('/')
def home():
	return render_template('main.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        age = int(request.form['age'])
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = request.form.get('fbs')
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = request.form.get('exang')
        oldpeak = float(request.form['oldpeak'])
        slope = request.form.get('slope')
        ca = int(request.form['ca'])
        thal = request.form.get('thal')
        
        data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        data2={"Age":age,"Sex":'Male' if sex == 1 else 'Female',"chest pain":cp,"Resting blood pressure":trestbps,"cholestrol":chol,"Resting electrocardiographic result":restecg,"Maximum heart rate":thalach,"Exercise induced angina":exang,"ST depression induced by exercise relative to rest":oldpeak," The slope of the peak exercise ST segment":slope,"Number of major vessels (0-3) colored by flourosopy":ca,"Reversable defect":thal}
        x=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
        y=[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]

        my_prediction = model.predict(data)
        
        return render_template('result.html', prediction=my_prediction,input=data2,labels=x,values=y)
     
      

        
        

if __name__ == '__main__':
	app.run(debug=True)

