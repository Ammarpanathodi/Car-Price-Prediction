from flask import Flask,request,app,url_for,render_template,jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
car = pd.read_csv("Cleaned_cars.csv")
pipe = pickle.load(open('linearregressionmodel.pkl','rb'))

@app.route('/')
def index():
    company = sorted(car['company'].unique())
    model = sorted(car['name'].unique())
    year = sorted(car['year'].unique())
    fuel = sorted(car['fuel_type'].unique())

    return render_template('index.html',company=company,model=model,year=year,fuel=fuel)



@app.route('/predict',methods=['POST'])
def prediction():
    company = request.form.get('company')
    model = request.form.get('model')
    year = request.form.get('year')
    fuel = request.form.get('fuel')
    km = request.form.get('km')
    print(company,model,year,fuel,km)

    input = pd.DataFrame([[model,company,year,km,fuel]],columns=['name','company','year','kms_driven','fuel_type'])
    pred = pipe.predict(input)
    print(pred[0])

    return  str(np.round(pred[0]))


if __name__=="__main__":
     app.run(debug=True,port=5004)
