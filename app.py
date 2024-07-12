from flask import Flask, render_template, request
import pickle
import numpy as np

model1 = pickle.load(open(r'"C:\Users\Keerthana\Desktop\solar panel\DA\solar power\svm_chennai.pkl"', 'rb'))  

app = Flask(__name__)  # initializing Flask app


@app.route("/",methods=['GET'])
def hello():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST': 

        d1 = request.form['year']
        d2 = request.form['month']
        d3 = request.form['date']
        d4 = request.form['time']
        d5 = request.form['temperature']
        d6 = request.form['relativehumidity']
        d7 = request.form['direct_radiation']
        d8 = request.form['diffuse_radiation']
        d9 = request.form['direct_normal_irradiance']
        d10 = request.form['windspeed']
        arr = np.array([[ d1,d2,d3,d4,d5,d6,d7,d8,d9,d10]])
        print([ d1,d2,d3,d4,d5,d6,d8,d9,d10])
        pred1 = model1.predict(arr)
        print(pred1)

    return render_template('result.html',prediction_text1=pred1)
    
if __name__ == '__main__':
    app.run(debug=False)
    
#app.run(host="0.0.0.0")            # deploy
            # run on local system
