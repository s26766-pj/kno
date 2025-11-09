import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical

# ============================================================
# 1. Wczytanie danych
# ============================================================

data_path = "data/wine.data"

columns = [
    "class", "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280/od315", "proline"
]

# Wczytanie CSV do DataFrame
df = pd.read_csv(data_path, header=None, names=columns)

# Oddzielenie cech (X) od etykiet (y)
X = df.drop("class", axis=1).values
y = df["class"].values - 1  # klasy 1,2,3 → 0,1,2

# One-Hot Encoding etykiet klas
y = to_categorical(y, num_classes=3)

# ============================================================
# 2. Podział danych i normalizacja
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# ============================================================
# 3. Model A – prosty MLP
# ============================================================

model1 = Sequential(name="Model_A_Basic")
model1.add(Dense(16, activation="relu", input_shape=(13,), name="Hidden_1"))
model1.add(Dense(12, activation="relu", name="Hidden_2"))
model1.add(Dense(3, activation="softmax", name="Output"))

# Kompilacja z Categorical Crossentropy
model1.compile(
    optimizer=Adam(learning_rate=0.1),
    loss=CategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

print("\n================== Model A Summary ==================")
model1.summary()

# Trenowanie Modelu 1
history1 = model1.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=250,
    batch_size=8,
    verbose=1
)

# ============================================================
# 4. Model B – głębszy MLP z BatchNorm i Dropout
# ============================================================

model2 = Sequential(name="Model_B_Advanced")
model2.add(Dense(64, activation="relu", input_shape=(13,), name="Hidden_1"))
model2.add(BatchNormalization(name="BatchNorm_1"))
model2.add(Dropout(0.2, name="Dropout_1"))

model2.add(Dense(32, activation="relu", name="Hidden_2"))
model2.add(BatchNormalization(name="BatchNorm_2"))
model2.add(Dropout(0.2, name="Dropout_2"))

model2.add(Dense(3, activation="softmax", name="Output"))

# Kompilacja
model2.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=CategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

print("\n================== Model 2 Summary ==================")
model2.summary()

# Trenowanie Modelu 2
history2 = model2.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=8,
    verbose=1
)

# ============================================================
# 5. Zapisywanie modeli i wyników
# ============================================================

# Tworzenie katalogów jeśli nie istnieją
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Zapisywanie modeli
model1.save("models/modelA.h5")
model2.save("models/modelB.h5")

print("\n✅ Modele zapisane w folderze: models/")

# ---------------------- WYKRESY - MODEL 1 ----------------------
plt.figure()
plt.plot(history1.history["accuracy"], label="train_accuracy")
plt.plot(history1.history["val_accuracy"], label="val_accuracy")
plt.title("Model A – Accuracy")
plt.xlabel("Epoka")
plt.ylabel("Dokładność")
plt.legend()
plt.savefig("results/modelA_learning_curves.png", dpi=200)
plt.close()

plt.figure()
plt.plot(history1.history["loss"], label="train_loss")
plt.plot(history1.history["val_loss"], label="val_loss")
plt.title("Model A – Loss")
plt.xlabel("Epoka")
plt.ylabel("Strata")
plt.legend()
plt.savefig("results/modelA_loss_curves.png", dpi=200)
plt.close()

# ---------------------- WYKRESY - MODEL 2 ----------------------
plt.figure()
plt.plot(history2.history["accuracy"], label="train_accuracy")
plt.plot(history2.history["val_accuracy"], label="val_accuracy")
plt.title("Model B – Accuracy")
plt.xlabel("Epoka")
plt.ylabel("Dokładność")
plt.legend()
plt.savefig("results/modelB_learning_curves.png", dpi=200)
plt.close()

plt.figure()
plt.plot(history2.history["loss"], label="train_loss")
plt.plot(history2.history["val_loss"], label="val_loss")
plt.title("Model B – Loss")
plt.xlabel("Epoka")
plt.ylabel("Strata")
plt.legend()
plt.savefig("results/modelB_loss_curves.png", dpi=200)
plt.close()

print("📁 Wykresy zapisane w folderze: results/")
print("\n✅ Trening zakończony pomyślnie!")
