import pandas as pd
import gradio as gr
import joblib
import numpy as np
import gradio as gr

features = ["GenHlth", "BMI", "DiffWalk", "Age", "Smoker", "Sex"]
pipeline = joblib.load("pipeline.joblib")
classes = ["Based on the assessment from the information you provided, we found that you have no risk of developing diabetes.", "Based on the assessment from the information you provided, we found that you have risk of developing diabetes."]

def predict(genhlth, diffWalk, age, smoker, sex, height, weight):
    height = float(height) / 100
    weight = float(weight)
    bmi = float(weight) / (float(height)**2)
    if (bmi >= 35):
        bmi = 5
    elif (bmi >= 30):
        bmi = 4
    elif (bmi >= 27):
        bmi = 3
    elif (bmi >= 24):
        bmi = 2
    elif (bmi <= 23):
        bmi = 1
    sample = dict()
    sample["GenHlth"] = genhlth
    sample["BMI"] = bmi
    sample["DiffWalk"] = diffWalk  
    sample["Age"] = age 
    sample["Smoker"] = smoker 
    sample["Sex"] = sex

    sample = pd.DataFrame([sample])
    y_pred = pipeline.predict_proba(sample)[0]
    y_pred = dict(zip(classes, y_pred)) 
    return y_pred

def cast_string_to_float(input):
    result = float(input)
    return result


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    options_genHealt = ["Excellent", "Very good", "Good", "Fair", "Poor"]
    values_genHealt = [1, 2, 3, 4, 5]
    options_age = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", ">80"]
    values_age = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    options_bmi = ["13-23", "24-26", "27-29", "30-34", ">35"]
    values_bmi = [1, 2, 3 ,4, 5]
    options_yes_no = ["Yes", "No"]
    values_yes_no = [1, 0]
    options_sex = ["Male", "Female"]
    values_sex = [1, 0]

    gr.Markdown("## Diabetes Indicator", elem_classes="markdown-label")
    gr.Markdown("""<h3>The dataset used for training the model comes from</h2> 
    <a href="https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset">https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset</a>""")
    gr.Markdown("""
        <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; padding: 15px; color: #721c24;">
            <h2 style="padding: 0; margin: 0; font-weight: bold; color: #ff5b4f;">Cautious</h2>
            <p style="padding: 0; margin: 10px 0 0; color: #721c24;">
                The result of the prediction is not a diagnosis, but only a prediction of the probability from the received data. If the prediction result shows a high probability of diabetes, a specialist should be contacted for a correct and accurate diagnosis.
            </p>
        </div>
    """)
    genhlth = gr.Radio(choices=list(zip(options_genHealt, values_genHealt)), label="General Health", info="Would you say that in general your health is", elem_classes="gr-radio")
    gr.Markdown("""
        <div style="display: block; background-color: #D3FFE1; border: 1px solid #D3FFE1; border-radius: 5px; padding: 15px; color: black;">
            <h3 style="padding: 0; margin: 0; font-weight: bold; color: #49A267;">Which general health should I choose?</h3>
            <h4 style="color: #333333;">Excellent</h4>
            <ul>
                <li style="color: #333333;">You feel very energetic and strong, with no health issues. You can perform physically demanding activities without any problems.</li>
                <li style="color: #333333;">You exercise regularly and maintain a healthy diet.</li>
                <li style="color: #333333;">You have never experienced any illness that interferes with your work or daily life.</li>
            </ul>
            <h4 style="color: #333333;">Very good</h4>
            <ul>
                <li style="color: #333333;">You generally feel healthy with no significant health concerns.</li>
                <li style="color: #333333;">You can carry out your daily activities, though you may occasionally feel tired.</li>
                <li style="color: #333333;">You try to exercise and eat well, but may have some inconsistencies.</li>
            </ul>
            <h4 style="color: #333333;">Good</h4>
            <ul>
                <li style="color: #333333;">You might experience minor health issues or not feel as energetic as before.</li>
                <li style="color: #333333;">You can perform daily activities, but sometimes feel tired or experience aches.</li>
                <li style="color: #333333;">You exercise and maintain your health at a moderate level, but not consistently.</li>
            </ul>
            <h4 style="color: #333333;">Fair</h4>
            <ul>
                <li style="color: #333333;">You have health issues that occasionally interfere with your daily activities.</li>
                <li style="color: #333333;">You often feel tired or have minor illnesses from time to time.</li>
                <li style="color: #333333;">You don't exercise regularly and may have unhealthy eating habits.</li>
            </ul>
            <h4 style="color: #333333;">Poor</h4>
            <ul>
                <li style="color: #333333;">You have significant health issues and feel unwell or fatigued most of the time.</li>
                <li style="color: #333333;">You have illnesses that disrupt your work or daily life.</li>
                <li style="color: #333333;">You don't exercise, and your diet is unhealthy.</li>
            </ul>
        </div>
    """)

    with gr.Row():
        smoker = gr.Radio(choices=list(zip(options_yes_no, values_yes_no)), label="Smoker", info="Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes]", elem_classes="gr-radio")
        sex = gr.Radio(choices=list(zip(options_sex, values_sex)), label="Sex", info="What is your sex?", elem_classes="gr-radio")
    
    with gr.Row():
        height = gr.Number(label="Height (cm)", info="What is your height? (centimeter)", value=1)
        weight = gr.Number(label="Weight (kg)", info="What is your weight? (kilogram)", value=1)
    age = gr.Radio(choices=list(zip(options_age, values_age)), label="Age", info="Select your age", elem_classes="gr-radio")
    diffWalk = gr.Radio(choices=list(zip(options_yes_no, values_yes_no)), label="Difficult walk", info="Do you have serious difficulty walking or climbing stairs?", elem_classes="gr-radio")

    predict_btn = gr.Button("Predict", variant="primary", elem_classes="gr-button-primary")
    Diabetes_binary = gr.Label(label="Diabetes_binary", elem_classes="gr-label")
    
    inputs = [genhlth, diffWalk, age, smoker, sex, height, weight]
    output = [Diabetes_binary]

    predict_btn.click(predict, inputs=inputs, outputs=output)

if __name__ == "__main__":
    demo.launch()
