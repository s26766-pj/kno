import argparse
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report


# ============================================================
# ARGPARSE
# ============================================================
parser = argparse.ArgumentParser(description="Train final model with given hyperparameters")
parser.add_argument("--params", type=str, default="models/best_params.json")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()


# ============================================================
# WCZYTANIE PARAMETRÓW
# ============================================================
with open(args.params, "r") as f:
    hp = json.load(f)


# ============================================================
# WCZYTANIE I PRZYGOTOWANIE DANYCH
# ============================================================
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

# ===================== SCALER =====================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# BUDOWA MODELU
# ============================================================
model = Sequential()

num_layers = hp["num_layers"]

for i in range(num_layers):
    units = hp[f"units_{i}"]
    model.add(Dense(units, activation="relu"))

    if hp.get(f"batchnorm_{i}", False):
        model.add(BatchNormalization())

    if hp.get(f"dropout_{i}", False):
        rate = hp.get(f"dropout_rate_{i}", 0.2)
        model.add(Dropout(rate))

model.add(Dense(3, activation="softmax"))

model.compile(
    optimizer=Adam(learning_rate=hp["lr"]),
    loss=CategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

print("\n===== Final Model Summary =====")
model.summary()


# ============================================================
# TRENING MODELU
# ============================================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=args.epochs,
    batch_size=args.batch_size,
    verbose=1
)


# ============================================================
# ZAPIS MODELU + SCALER
# ============================================================
model.save("models/final_model.h5")
joblib.dump(scaler, "models/scaler.save")

print("\n====================================")
print(" Finalny model zapisany: models/final_model.h5")
print(" Scaler zapisany: models/scaler.save")
print("====================================")


# ============================================================
# GENEROWANIE WYKRESÓW
# ============================================================

# ------ ACCURACY ------
plt.figure()
plt.plot(history.history["accuracy"], label="train_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.title("Final Model – Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("results/final_model_accuracy.png", dpi=200)
plt.close()

# ------ LOSS ------
plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Final Model – Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("results/final_model_loss.png", dpi=200)
plt.close()

print("📁 Wykresy zapisane w folderze: results/")
print("✅ Trening zakończony pomyślnie!\n")

print("\n===== Final Model Summary =====")
model.summary()

# =================== Ewaluacja ===================
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

acc = np.mean(y_pred == y_true)
print(f"\nDokładność na zbiorze testowym: {acc:.4f}\n")

# Macierz pomyłek
cm = confusion_matrix(y_true, y_pred)
print("Macierz pomyłek (Confusion Matrix):")
print(cm)

# Szczegółowy raport klasyfikacji
report = classification_report(y_true, y_pred, target_names=[f"Class {i}" for i in range(3)])
print("\nRaport klasyfikacji:")
print(report)
