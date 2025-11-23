import argparse
import json
import os

import pandas as pd


# biblioteki do ML i preprocessing danych
from sklearn.model_selection import train_test_split  # do podziału danych na zbiór treningowy i testowy
from sklearn.preprocessing import StandardScaler      # do standaryzacji cech (normalizacja)

# Keras / TensorFlow – sieci neuronowe
from tensorflow.keras.models import Sequential          # prosty model sieci warstwowej
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical     # do konwersji etykiet do postaci one-hot

# Keras Tuner – do automatycznego strojenia hiperparametrów
import keras_tuner as kt


# =====================================================================
# 1. ARGPARSE – parametry do tunera
# =====================================================================
# Tworzymy parser argumentów dla skryptu, aby można było ustawiać parametry z linii poleceń
parser = argparse.ArgumentParser(description="Hyperparameter search with Keras Tuner")

# maksymalna liczba epok treningowych
parser.add_argument("--max_epochs", type=int, default=20)

# ile razy każda konfiguracja modelu będzie trenowana (przydatne przy losowości)
parser.add_argument("--executions", type=int, default=1)

# parsowanie argumentów wejściowych
args = parser.parse_args()


# =====================================================================
# 2. Wczytanie danych
# =====================================================================
data_path = "data/wine.data"  # ścieżka do pliku CSV z danymi win

# nazwy kolumn zgodne z dokumentacją zbioru danych Wine
columns = [
    "class", "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280/od315", "proline"
]

# wczytanie danych do pandas DataFrame
df = pd.read_csv(data_path, header=None, names=columns)

# oddzielenie cech (X) od etykiet klas (y)
X = df.drop("class", axis=1).values  # wszystkie kolumny oprócz "class"
y = df["class"].values - 1           # klasy zaczynają się od 1, przeskalowanie do 0-index

# konwersja etykiet do one-hot encoding (np. 3 klasy -> [1,0,0], [0,1,0], [0,0,1])
y = to_categorical(y, num_classes=3)

# podział na zbiór treningowy i testowy (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# standaryzacja cech: mean=0, std=1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # dopasowanie scaler i transformacja zbioru treningowego
X_test = scaler.transform(X_test)        # transformacja zbioru testowego na podstawie scaler z treningu


# =====================================================================
# 3. Funkcja budująca model dla tunera
# =====================================================================
def build_model(hp):
    """
    Funkcja tworzy model Keras dla tunera hiperparametrów.
    hp: obiekt Keras Tuner do definiowania zakresów hiperparametrów.
    """
    model = Sequential()  # tworzymy model warstwowy "sequential"

    # liczba warstw ukrytych do strojenia (1-4)
    num_layers = hp.Int("num_layers", min_value=1, max_value=4)

    # iteracja po warstwach ukrytych
    for i in range(num_layers):
        # liczba neuronów w warstwie ukrytej do strojenia (8-128)
        model.add(
            Dense(
                hp.Int(f"units_{i}", min_value=8, max_value=128, step=8),
                activation="relu"
            )
        )

        # opcjonalne dodanie BatchNormalization
        if hp.Boolean(f"batchnorm_{i}"):
            model.add(BatchNormalization())

        # opcjonalne dodanie Dropout
        if hp.Boolean(f"dropout_{i}"):
            model.add(Dropout(rate=hp.Float(f"dropout_rate_{i}", 0.1, 0.5, step=0.1)))

    # warstwa wyjściowa (3 klasy, softmax)
    model.add(Dense(3, activation="softmax"))

    # learning rate do strojenia
    lr = hp.Float("lr", 1e-4, 1e-1, sampling="log")

    # kompilacja modelu
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=CategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )

    return model


# =====================================================================
# 4. Konfiguracja tunera
# =====================================================================
# RandomSearch – tuner losowy (przeszukuje losowo przestrzeń hiperparametrów)
tuner = kt.RandomSearch(
    build_model,                 # funkcja budująca model
    objective="val_accuracy",    # cel optymalizacji – maksymalizacja dokładności na zbiorze walidacyjnym
    max_trials=20,               # maksymalna liczba różnych konfiguracji do sprawdzenia
    executions_per_trial=args.executions,  # ile razy powtarzać każdą konfigurację
    directory="tuner_results",   # katalog do zapisu wyników tuningu
    project_name="wine_tuning"   # nazwa projektu
)

# wyszukiwanie najlepszych hiperparametrów
tuner.search(
    X_train, y_train,                # dane treningowe
    validation_data=(X_test, y_test),# dane walidacyjne
    epochs=args.max_epochs,          # liczba epok treningowych
    verbose=1                        # poziom szczegółowości logów
)

# pobranie najlepszych hiperparametrów
best_hp = tuner.get_best_hyperparameters(1)[0]

# zapis najlepszych parametrów do pliku JSON
os.makedirs("models", exist_ok=True)   # tworzymy katalog jeśli nie istnieje
with open("models/best_params.json", "w") as f:
    json.dump(best_hp.values, f, indent=4)

# informacja końcowa
print("\n====================================")
print(" Najlepsze hiperparametry zapisano do models/best_params.json")
print("====================================")
