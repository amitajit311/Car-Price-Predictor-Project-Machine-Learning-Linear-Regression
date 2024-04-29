from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np
app = Flask(__name__)   # Flask constructor
# cors=CORS(app)

model=pickle.load(open('LinearRegressionModel.pkl','rb'))
car = pd.read_csv('Cleaned Car.csv')

# A decorator used to tell the application
# which URL is associated function
@app.route('/',methods=['GET','POST'])     
def hello():
    compaines = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year =  sorted(car['year'].unique(),reverse=True)
    fuel_type = car['fuel_type'].unique()
    compaines.insert(0,"Select Company")
    return render_template('index.html',compaines=compaines,car_models=car_models,years=year,fuel_type=fuel_type)


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    company=request.form.get('company')
    car_model=request.form.get('car_model')
    year=int(request.form.get('year'))
    fuel_type=request.form.get('fuel_type')
    kms_driven=int(request.form.get('kilo_driven'))
    print(company,car_model,year,fuel_type,kms_driven)
    prediction = model.predict(pd.DataFrame([[car_model,company,year,kms_driven,fuel_type]],columns=['name','company','year','kms_driven','fuel_type']))
    print(prediction[0])


    return str(np.round(prediction[0],2))
  
if __name__=='__main__':
   app.run(debug=True)