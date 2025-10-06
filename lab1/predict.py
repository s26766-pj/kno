import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from ui import create_ui

class DiabetesPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model_and_scaler()
    
    def load_model_and_scaler(self):
        """Ładuje model i scaler z plików"""
        try:
            # Ładowanie modelu
            model_path = os.path.join(os.path.dirname(__file__), "trained_model_for_diabetes_prediction.keras")
            self.model = keras.models.load_model(model_path)
            
            # Ładowanie danych treningowych i tworzenie scalera
            self.scaler = StandardScaler()
            data_path = os.path.join(os.path.dirname(__file__), "data", "diabetes.csv")
            df_train = pd.read_csv(data_path)
            self.scaler.fit(df_train.drop("Outcome", axis=1).values)
            
            print("Model i scaler załadowane pomyślnie!")
            
        except Exception as e:
            print(f"Błąd ładowania modelu: {str(e)}")
            raise e
    
    def predict(self, patient_data):
        """Wykonuje predykcję na podstawie danych pacjenta"""
        try:
            # Przekształć dane używając scalera
            patient_scaled = self.scaler.transform([patient_data])
            
            # Wykonaj predykcję
            prediction = self.model.predict(patient_scaled, verbose=0)
            probability = prediction[0][0]
            
            # Formatuj wynik
            result = f"=== DIABETES PREDICTION RESULT ===\n\n"
            result += f"Probability of Diabetes: {probability:.4f} ({probability*100:.2f}%)\n\n"
            
            if probability >= 0.5:
                result += "🔴 HIGH RISK: Patient is likely to have diabetes.\n"
                result += "Recommendation: Consult with a healthcare professional immediately."
            else:
                result += "🟢 LOW RISK: Patient is likely to be diabetes-free.\n"
                result += "Recommendation: Continue maintaining a healthy lifestyle."
            
            return result
            
        except Exception as e:
            raise Exception(f"Błąd podczas predykcji: {str(e)}")

def main():
    """Główna funkcja uruchamiająca aplikację"""
    try:
        # Utwórz predictor
        predictor = DiabetesPredictor()
        
        # Funkcja callback dla UI
        def prediction_callback(patient_data):
            return predictor.predict(patient_data)
        
        # Utwórz i uruchom UI
        root = create_ui(prediction_callback)
        print("Uruchamianie interfejsu użytkownika do predykcji cukrzycy...")
        root.mainloop()
        
    except Exception as e:
        print(f"Błąd uruchamiania aplikacji: {str(e)}")

if __name__ == "__main__":
    main()