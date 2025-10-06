import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# Wczytanie zapisanego modelu
model = keras.models.load_model("trained_model_for_diabetes_prediction.keras")

# Dane pacjenta w tej samej kolejności co w CSV:
# Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
patient_data = [0, 130, 90, 30, 100, 25.0, 0.5, 35]  # przykładowe wartości


scaler = StandardScaler()
df_train = pd.read_csv("data/diabetes.csv")
scaler.fit(df_train.drop("Outcome", axis=1).values)

# Przekształcamy dane pacjenta
patient_scaled = scaler.transform([patient_data])  # zamieniamy listę na tablicę 2D

# Predykcja
prediction = model.predict(patient_scaled)

# Interpretacja
print("Prawdopodobieństwo cukrzycy:", prediction[0][0])
if prediction[0][0] >= 0.5:
    print("Model przewiduje: Pacjent prawdopodobnie ma cukrzycę.")
else:
    print("Model przewiduje: Pacjent prawdopodobnie nie ma cukrzycy.")
