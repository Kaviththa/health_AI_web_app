from flask import Flask,render_template, request
import requests
import pandas as pd
import numpy as np
import pickle



app = Flask(__name__)
model = pickle.load(open('model/diabetes_log_model_5.pkl', 'rb'))
scaler = pickle.load(open('model/minmax_scaler_5.pkl', 'rb'))

model1 = pickle.load(open('model/heart_model5.pkl', 'rb'))

model_kidney= pickle.load(open('model/kidney_svm_model-5.pkl', 'rb'))
scaler_kidney = pickle.load(open('model/minmax_scaler-5.pkl', 'rb'))

model_liver = pickle.load(open('model/liver_svm_model.pkl', 'rb'))
scaler_liver = pickle.load(open('model/liver_standard_scaler.pkl', 'rb'))

@app.route('/',methods=['GET'])
def index():
    return  render_template('index.html')

@app.route('/diabetes',methods=['GET'])
def diabetes():
    return  render_template('diabetes.html')


@app.route("/predict", methods=['POST'])
def predict():
      if request.method == 'POST':
          Pregnancies = float(request.form['Pregnancies'])
          Glucose = float(request.form['Glucose'])
          BloodPressure = float(request.form['BloodPressure'])
          SkinThinkness= float(request.form['SkinThinkness'])
          Insulin = float(request.form['Insulin'])
          BMI = float(request.form['BMI'])
          DPF= float(request.form['DPF'])
          Age = int(request.form['Age'])
          
          prediction = model.predict(scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThinkness,Insulin,BMI,DPF,Age]]))[0]
          prediction = int(prediction)
          return render_template('diabetes.html',prediction_text=prediction)
          
          
    
@app.route('/heart',methods=['GET'])
def heart():
    return  render_template('heart.html')


@app.route("/predict_heart", methods=['POST'])
def predict_heart():
      if request.method == 'POST':
          age = int(request.form['age'])
          sex = int(request.form['sex'])
          cp = int(request.form['cp'])
          trestbps= int(request.form['trestbps'])
          chol = int(request.form['chol'])
          fbs = int(request.form['fbs'])
          restecg= int(request.form['restecg'])
          thalach = int(request.form['thalach'])
          exang = int(request.form['exang'])
          oldpeak = float(request.form['oldpeak'])
          slope = int(request.form['slope'])
          ca = int(request.form['ca'])
          thal= int(request.form['thal'])
          
          
          prediction=model1.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
          prediction = round(prediction[0],2)
          
          return render_template('heart.html',heart_result=prediction)
          
             
          
            
@app.route('/kidney',methods=['GET'])
def kidney():
    return  render_template('kidney.html')



@app.route('/kidney_predict',methods=['POST'])
def kidney_predict():
    if request.method == 'POST':
          age = float(request.form['age'])
          bp = float(request.form['bp'])
          al = float(request.form['al'])
          su= float(request.form['su'])
          pcc = int(request.form['pcc'])
          ba= int(request.form['ba'])
          bgr = float(request.form['bgr'])
          bu = float(request.form['bu'])
          sc = float(request.form['sc'])
          sod = float(request.form['sod'])
          pot = float(request.form['pot'])
          hemo= float(request.form['hemo'])
          pcv = float(request.form['pcv'])
          wc = float(request.form['wc'])
          htn= int(request.form['htn'])
          dm = int(request.form['dm'])
          cad = int(request.form['cad'])
          appet= int(request.form['appet'])
          pe = int(request.form['pe'])
          ane = int(request.form['ane'])
            
          
         
          
          bp =bp/100
          
          X =[age,bp,al,su,pcc,ba,bgr,bu,sc,sod, pot,hemo,pcv,wc,htn,dm,cad,appet,pe,ane]
          y = pd.DataFrame([X],columns=['age','bp','al','su','pcc','ba','bgr','bu','sc','sod', 'pot','hemo','pcv','wc','htn','dm','cad','appet','pe','ane'])
          #dataset_to_scaled = [age,bgr,bu,sc,sod,pot,hemo,pcv,wc]
          y[['age','bgr','bu','sc','sod','pot','hemo','pcv','wc']]= pd.DataFrame(scaler_kidney.transform(y[['age','bgr','bu','sc','sod','pot','hemo','pcv','wc']]))
          
          prediction = model_kidney.predict(y)[0]
          prediction = int(prediction)
          
          
          return  render_template('kidney.html', kidney_result=prediction)
      
        
      
@app.route('/liver',methods=['GET'])
def liver():
    return  render_template('liver.html')


@app.route('/liver_predict',methods=['POST'])
def liver_predict():
    if request.method == 'POST':
          Age = int(request.form['Age'])
          Gender = int(request.form['Gender'])
          Total_Bilirubin = float(request.form['Total_Bilirubin'])
          Alkaline_Phosphotase= float(request.form['Alkaline_Phosphotase'])
          Alamine_Aminotransferase  = float(request.form['Alamine_Aminotransferase'])
          Aspartate_Aminotransferase= float(request.form['Aspartate_Aminotransferase'])
          Total_Protiens = float(request.form['Total_Protiens'])
          Albumin  = float(request.form['Albumin'])
          Albumin_and_Globulin_Ratio = float(request.form['Albumin_and_Globulin_Ratio'])
          
            
          
         
          
         
          
          prediction_liver = model_liver.predict(scaler_liver.transform([[Age,Gender,Total_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]]))[0]
         
          
          prediction = int(prediction_liver)
          
          
          return  render_template('liver.html', liver_result=prediction)
      
        
@app.route('/about',methods=['GET'])
def about():
    return  render_template('about.html')
      
      
if __name__=="__main__":
    app.run(debug=True)



