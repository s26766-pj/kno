import argparse
import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical

import keras_tuner as kt


# =====================================================================
# 1. ARGPARSE – parametry do tunera (np. max_epochs)
# =====================================================================
parser = argparse.ArgumentParser(description="Hyperparameter search with Keras Tuner")
parser.add_argument("--max_epochs", type=int, default=20)
parser.add_argument("--executions", type=int, default=1)
args = parser.parse_args()


# =====================================================================
# 2. Wczytanie danych
# =====================================================================
data_path = "data/wine.data"
columns = [
    "class", "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280/od315", "proline"
]

df = pd.read_csv(data_path, header=None, names=columns)

X = df.drop("class", axis=1).values
y = df["class"].values - 1
y = to_categorical(y, num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =====================================================================
# 3. Funkcja budująca model dla tunera
# =====================================================================
def build_model(hp):
    model = Sequential()

    # liczba warstw ukrytych
    num_layers = hp.Int("num_layers", min_value=1, max_value=4)

    for i in range(num_layers):
        model.add(
            Dense(
                hp.Int(f"units_{i}", min_value=8, max_value=128, step=8),
                activation="relu"
            )
        )

        if hp.Boolean(f"batchnorm_{i}"):
            model.add(BatchNormalization())

        if hp.Boolean(f"dropout_{i}"):
            model.add(Dropout(rate=hp.Float(f"dropout_rate_{i}", 0.1, 0.5, step=0.1)))

    # output
    model.add(Dense(3, activation="softmax"))

    # optimizer
    lr = hp.Float("lr", 1e-4, 1e-1, sampling="log")

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=CategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )

    return model


# =====================================================================
# 4. Konfiguracja tunera
# =====================================================================
tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=20,
    executions_per_trial=args.executions,
    directory="tuner_results",
    project_name="wine_tuning"
)

tuner.search(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=args.max_epochs,
    verbose=1
)

# najlepsze parametry
best_hp = tuner.get_best_hyperparameters(1)[0]

# zapis do JSON
os.makedirs("models", exist_ok=True)
with open("models/best_params.json", "w") as f:
    json.dump(best_hp.values, f, indent=4)

print("\n====================================")
print(" Najlepsze hiperparametry zapisano do models/best_params.json")
print("====================================")
