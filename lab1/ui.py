import tkinter as tk
from tkinter import ttk, messagebox

class DiabetesPredictionUI:
    def __init__(self, root, prediction_callback=None):
        self.root = root
        self.root.title("Diabetes Prediction Tool")
        self.root.geometry("600x600")
        self.root.resizable(False, False)
        self.prediction_callback = prediction_callback
        
        # Utwórz interfejs
        self.create_widgets()
    
    def create_widgets(self):
        """Tworzy elementy interfejsu"""
        # Tytuł
        title_label = tk.Label(self.root, text="Diabetes Prediction Tool", 
                              font=("Arial", 16, "bold"), fg="darkblue")
        title_label.pack(pady=10)
        
        # Ramka dla pól wejściowych
        input_frame = ttk.LabelFrame(self.root, text="Patient Data", padding=20)
        input_frame.pack(pady=10, padx=20, fill="x")
        
        # Lista pól i ich opisów
        self.fields = [
            ("Pregnancies", "Number of pregnancies"),
            ("Glucose", "Plasma glucose concentration (mg/dL)"),
            ("BloodPressure", "Diastolic blood pressure (mm Hg)"),
            ("SkinThickness", "Triceps skin fold thickness (mm)"),
            ("Insulin", "2-Hour serum insulin (mu U/ml)"),
            ("BMI", "Body mass index (kg/m²)"),
            ("DiabetesPedigreeFunction", "Diabetes pedigree function"),
            ("Age", "Age (years)")
        ]
        
        self.entries = {}
        
        # Tworzenie pól wejściowych
        for i, (field, description) in enumerate(self.fields):
            # Label z opisem
            label = tk.Label(input_frame, text=f"{field}:", font=("Arial", 10, "bold"))
            label.grid(row=i, column=0, sticky="w", pady=2)
            
            # Pole wejściowe
            entry = tk.Entry(input_frame, width=15, font=("Arial", 10))
            entry.grid(row=i, column=1, padx=(10, 0), pady=2, sticky="w")
            self.entries[field] = entry
            
            # Opis pola
            desc_label = tk.Label(input_frame, text=description, 
                                font=("Arial", 8), fg="gray")
            desc_label.grid(row=i, column=2, padx=(10, 0), pady=2, sticky="w")
        
        # Wypełnij przykładowymi danymi
        self.fill_example_data()
        
        # Ramka dla przycisków
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)
        
        # Przycisk predykcji
        predict_button = tk.Button(button_frame, text="Predict Diabetes Risk", 
                                  command=self.predict_diabetes,
                                  bg="lightblue", font=("Arial", 12, "bold"),
                                  padx=20, pady=10)
        predict_button.pack(side=tk.LEFT, padx=5)
        
        # Przycisk czyszczenia
        clear_button = tk.Button(button_frame, text="Clear", 
                                command=self.clear_fields,
                                bg="lightgray", font=("Arial", 12),
                                padx=20, pady=10)
        clear_button.pack(side=tk.LEFT, padx=5)
        
        # Ramka dla wyników
        result_frame = ttk.LabelFrame(self.root, text="Prediction Result", padding=20)
        result_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        # Pole tekstowe dla wyników
        self.result_text = tk.Text(result_frame, height=8, width=50, 
                                  font=("Arial", 11), wrap=tk.WORD)
        self.result_text.pack(fill="both", expand=True)
        
        # Scrollbar dla pola tekstowego
        scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=self.result_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.result_text.configure(yscrollcommand=scrollbar.set)
    
    def fill_example_data(self):
        """Wypełnia pola przykładowymi danymi"""
        example_data = {
            "Pregnancies": "0",
            "Glucose": "130",
            "BloodPressure": "90",
            "SkinThickness": "30",
            "Insulin": "100",
            "BMI": "25.0",
            "DiabetesPedigreeFunction": "0.5",
            "Age": "35"
        }
        
        for field, value in example_data.items():
            self.entries[field].insert(0, value)
    
    def clear_fields(self):
        """Czyści wszystkie pola wejściowe"""
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.result_text.delete(1.0, tk.END)
    
    def validate_input(self):
        """Waliduje dane wejściowe"""
        patient_data = []
        
        for field, entry in self.entries.items():
            try:
                value = float(entry.get())
                if field in ["Pregnancies", "Age"] and value < 0:
                    raise ValueError(f"{field} nie może być ujemne")
                if field == "BMI" and value <= 0:
                    raise ValueError("BMI musi być większe od 0")
                patient_data.append(value)
            except ValueError as e:
                messagebox.showerror("Błąd walidacji", 
                                   f"Nieprawidłowa wartość w polu {field}: {str(e)}")
                return None
        
        return patient_data
    
    def predict_diabetes(self):
        """Wykonuje predykcję cukrzycy"""
        # Waliduj dane wejściowe
        patient_data = self.validate_input()
        if patient_data is None:
            return
        
        if self.prediction_callback:
            # Użyj callback do predykcji
            try:
                result = self.prediction_callback(patient_data)
                self.display_result(result, patient_data)
            except Exception as e:
                messagebox.showerror("Błąd predykcji", f"Wystąpił błąd podczas predykcji: {str(e)}")
        else:
            messagebox.showerror("Błąd", "Brak funkcji predykcji")
    
    def display_result(self, result_text, patient_data):
        """Wyświetla wynik predykcji"""
        # Wyczyść poprzednie wyniki
        self.result_text.delete(1.0, tk.END)
        
        # Dodaj dane pacjenta do wyniku
        full_result = result_text + f"\n\n=== PATIENT DATA ===\n"
        for i, (field, _) in enumerate(self.fields):
            full_result += f"{field}: {patient_data[i]}\n"
        
        self.result_text.insert(1.0, full_result)

def create_ui(prediction_callback=None):
    """Tworzy i zwraca instancję UI"""
    root = tk.Tk()
    app = DiabetesPredictionUI(root, prediction_callback)
    
    # Centruj okno na ekranie
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    return root

def main():
    """Główna funkcja uruchamiająca aplikację (bez callback)"""
    root = create_ui()
    root.mainloop()

if __name__ == "__main__":
    main()
