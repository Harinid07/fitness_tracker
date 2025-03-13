import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
import warnings

warnings.filterwarnings("ignore")

# Custom CSS for styling
st.markdown(
    """
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stProgress > div > div > div > div {
            background-color: #4CAF50;
        }
        .header-title {
            font-size: 32px;
            font-weight: bold;
            color: #2c3e50;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and introduction
st.markdown('<p class="header-title">🏋️‍♂️ Personal Fitness Tracker</p>', unsafe_allow_html=True)
st.write(
    "Welcome to the Personal Fitness Tracker! This app predicts how many kilocalories "
    "you will burn based on your age, BMI, exercise duration, heart rate, and body temperature."
)

# Sidebar for user inputs
st.sidebar.header("🔹 User Input Parameters")

def user_input_features():
    age = st.sidebar.slider("Age", 10, 100, 30)
    bmi = st.sidebar.slider("BMI", 15, 40, 20)
    duration = st.sidebar.slider("Duration (min)", 0, 35, 15)
    heart_rate = st.sidebar.slider("Heart Rate", 60, 130, 80)
    body_temp = st.sidebar.slider("Body Temperature (°C)", 36, 42, 38)
    gender_button = st.sidebar.radio("Gender", ("Male", "Female"))
    gender = 1 if gender_button == "Male" else 0
    return pd.DataFrame({
        "Age": [age], "BMI": [bmi], "Duration": [duration],
        "Heart_Rate": [heart_rate], "Body_Temp": [body_temp], "Gender_male": [gender]
    })

df = user_input_features()

# Predict button
if st.sidebar.button("Predict Calories Burned"):
    # Display user inputs
    st.write("## 📝 Your Entered Parameters:")
    st.dataframe(df, width=600)

    # Load dataset
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")
    exercise_df = exercise.merge(calories, on="User_ID").drop(columns="User_ID")
    exercise_df["BMI"] = round(exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2), 2)
    exercise_train, exercise_test = train_test_split(exercise_df, test_size=0.2, random_state=1)

    exercise_train = exercise_train[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
    exercise_test = exercise_test[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]

    exercise_train = pd.get_dummies(exercise_train, drop_first=True)
    exercise_test = pd.get_dummies(exercise_test, drop_first=True)

    X_train, y_train = exercise_train.drop("Calories", axis=1), exercise_train["Calories"]
    X_test, y_test = exercise_test.drop("Calories", axis=1), exercise_test["Calories"]

    # Train model
    model = RandomForestRegressor(n_estimators=1000, max_depth=6, max_features=3)
    model.fit(X_train, y_train)

    df = df.reindex(columns=X_train.columns, fill_value=0)
    prediction = model.predict(df)

    # Display predictions with animations
    st.write("## 🔥 Predicted Calories Burned")
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.01)
    st.success(f"You will burn approximately **{round(prediction[0], 2)} kilocalories**!")

    # Show similar results
    st.write("## 🔍 Similar Calorie Burn Data")
    similar_data = exercise_df[(exercise_df["Calories"] >= prediction[0] - 10) & (exercise_df["Calories"] <= prediction[0] + 10)]
    st.dataframe(similar_data.sample(5))

    # Show statistical insights
    st.write("## 📊 Insights on Your Inputs")
    st.write(f"You are older than **{round((exercise_df['Age'] < df['Age'].values[0]).mean() * 100, 2)}%** of users.")
    st.write(f"Your exercise duration is longer than **{round((exercise_df['Duration'] < df['Duration'].values[0]).mean() * 100, 2)}%** of users.")
    st.write(f"Your heart rate is higher than **{round((exercise_df['Heart_Rate'] < df['Heart_Rate'].values[0]).mean() * 100, 2)}%** of users.")
    st.write(f"Your body temperature is higher than **{round((exercise_df['Body_Temp'] < df['Body_Temp'].values[0]).mean() * 100, 2)}%** of users.")

    # Personalized Diet Plan
    st.write("## 🍽️ Personalized Diet Plan")
    if df["BMI"].values[0] < 18.5:
        st.info("Your BMI suggests you may need a **high-protein diet** to gain healthy weight. Include eggs, lean meats, nuts, and dairy.")
    elif 18.5 <= df["BMI"].values[0] <= 24.9:
        st.success("You have a **balanced BMI**. Maintain a well-rounded diet with fruits, vegetables, lean proteins, and whole grains.")
    else:
        st.warning("Your BMI suggests a **weight-loss-friendly diet** may be beneficial. Focus on fiber-rich foods, lean protein, and avoid high-sugar foods.")

    if df["Duration"].values[0] > 20:
        st.write("Since you exercise for **over 20 minutes**, consider **post-workout meals** like bananas, protein shakes, or nuts.")

    st.write("---")
    st.write("### 📌 Stay consistent and track your fitness journey regularly! 🚀")


