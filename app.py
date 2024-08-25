import pandas as pd
import gradio as gr
import joblib
import numpy as np

features = ["GenHlth", "BMI", "HeartDiseaseorAttack", "DiffWalk", "HvyAlcoholConsump"]
pipeline = joblib.load("pipeline.joblib")
classes = ["นายรอดนายไม่เป็นโรคนะจ้ะ", "ว้ายยนายมีความเสี่ยงไปหาหมอซะนะ"]
def predict(genhlth, highbp, highchol, bmi, age, diffWalk):
  sample = dict()
  sample["GenHlth"] = genhlth
  sample["HighBP"] = highbp
  sample["HighChol"] = highchol
  sample["BMI"] = bmi
  sample["Age"] = age
  sample["DiffWalk"] = diffWalk

  sample = pd.DataFrame([sample])
  y_pred = pipeline.predict_proba(sample)[0]
  y_pred = dict(zip(classes, y_pred)) 
  return y_pred

with gr.Blocks() as demo:
  options_age = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", ">80"]
  values_age = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
  options_bmi = ["13-23", "24-26", "27-29", "30-34", ">35"]
  values_bmi = [1, 2, 3 ,4, 5]
  options_yes_no = ["Yes", "No"]
  values_yes_no = [1, 0]
  genhlth = gr.Slider(value=1, label="General Health", info="Would you say that in general your health is: scale 1-5", minimum=1, maximum=5, step=1, interactive=True)
  highbp = gr.Radio(choices=list(zip(options_yes_no, values_yes_no)), label="High Blood Pressure", info="Do you have High Blood Pressure?")
  highchol = gr.Radio(choices=list(zip(options_yes_no, values_yes_no)), label="High Cholesterol ", info="Do you have high cholesterol?")
  bmi = gr.Radio(choices=list(zip(options_bmi, values_bmi)), label="BMI", info="Body Mass Index")
  age = gr.Radio(choices=list(zip(options_age, values_age)), label="Age", info="Choose your age")
  diffWalk = gr.Radio(choices=list(zip(options_yes_no, values_yes_no)), label="Difficult walk", info="Do you have serious difficulty walking or climbing stairs?")

  predict_btn = gr.Button("Predict", variant="primary")
  Diabetes_binary = gr.Label(label="Diabetes_binary")
  
  inputs = [genhlth, highbp, highchol, bmi, age, diffWalk]
  output = [Diabetes_binary]

  predict_btn.click(predict, inputs=inputs, outputs=output)
if __name__ == "__main__":
  demo.launch()
