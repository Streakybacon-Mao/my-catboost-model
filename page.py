import streamlit as st
import pandas as pd
import pickle
from catboost import CatBoostClassifier

# 加载模型
with open('catboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 创建UI
st.title("Model Prediction of depression in diabetes patients")

# 定义变量
continuous_vars = ['BMI', 'Age', 'sleep', 'HEI2020', 'PIR', 'HDL_C', 'Triglycerides', 'Cholesterol']
categorical_vars = ['Alcohol', 'Hypertension', 'gender', 'Race/Ethnicity',
                    'Education', 'Marital_Status', 'smoke', 'PA', 'Chest_pain']

# 连续变量的取值范围
continuous_ranges = {
    'BMI': (15, 50),
    'Age': (0, 100),
    'sleep': (0, 20),
    'HEI2020': (0, 100),
    'PIR': (0, 5),
    'HDL_C': (10, 160),
    'Triglycerides': (0, 50),
    'Cholesterol': (2, 16)
}

# 连续变量输入
st.header("Continuous Variables")
continuous_input = {}
for var in continuous_vars:
    min_val, max_val = continuous_ranges[var]
    continuous_input[var] = st.slider(var, min_value=min_val, max_value=max_val, value=(min_val + max_val) // 2)

# 分类变量输入
st.header("Categorical Variables")
categorical_input = {}

# 特殊处理的分类变量
special_vars = {
    'Alcohol': {1: 'Yes', 2: 'No'},
    'Hypertension': {1: 'Yes', 2: 'No'},
    'PA': {1: 'Yes', 2: 'No'},
    'Chest_pain': {1: 'Yes', 2: 'No'},
    'smoke': {1: 'Yes', 2: 'No'},
    'gender': {1: 'Male', 2: 'Female'},
    'Race/Ethnicity': {
        1: 'Mexican American',
        2: 'Other Hispanic',
        3: 'Non-Hispanic White',
        4: 'Non-Hispanic Black',
        5: 'Other Race - Including Multi-Racial'
    },
    'Education': {
        1: 'Less Than 9th Grade',
        2: '9-11th Grade (Includes 12th grade with no diploma)',
        3: 'High School Grad/GED or Equivalent',
        4: 'Some College or AA degree',
        5: 'College Graduate or above'
    },
    'Marital_Status': {
        1: 'Married',
        2: 'Widowed',
        3: 'Divorced',
        4: 'Separated',
        5: 'Never married',
        6: 'Living with partner'
    }
}

# 处理特殊分类变量
for var in special_vars:
    options = list(special_vars[var].values())
    selected = st.selectbox(var, options)
    # 将文本映射回数值
    categorical_input[var] = [k for k, v in special_vars[var].items() if v == selected][0]

# 将输入转换为DataFrame
input_data = {**continuous_input, **categorical_input}
input_df = pd.DataFrame([input_data])

# 确保输入数据的列顺序与训练时的特征顺序一致
training_features = [
    'Alcohol', 'BMI', 'Hypertension', 'gender', 'Race/Ethnicity', 'Education',
    'Marital_Status', 'Age', 'sleep', 'smoke', 'HEI2020', 'PIR', 'PA',
    'Cholesterol', 'Chest_pain', 'HDL_C', 'Triglycerides'
]

# 检查并重新排列列顺序
input_df = input_df[training_features]

# 预测
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write(f"Prediction: {prediction[0]}")

# 可选：显示输入数据
st.subheader("Input Data")
st.write(input_df)