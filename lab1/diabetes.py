# Importujemy potrzebne biblioteki
import pandas as pd                    # do pracy z plikami CSV (tabele danych)
import matplotlib.pyplot as plt        # do rysowania wykresów
import tensorflow
from sklearn.model_selection import train_test_split   # do podziału danych na trening/test
from sklearn.preprocessing import StandardScaler       # do normalizacji danych
from tensorflow import keras


# Funkcja do trenowania modelu
def train_model(epochs=50, batch_size=32, learning_rate=0.001):
    # 1. Wczytanie danych
    # Otwieramy plik CSV z danymi o cukrzycy
    df = pd.read_csv("data/diabetes.csv")

    # X = dane wejściowe (wszystkie kolumny oprócz "Outcome")
    X = df.drop("Outcome", axis=1).values
    # y = etykiety (czy osoba ma cukrzycę: 0 = nie, 1 = tak)
    y = df["Outcome"].values

    # 2. Standaryzacja danych
    # Skalujemy dane, żeby wszystkie cechy miały podobny zakres (średnia=0, odchylenie=1).
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 3. Podział danych na treningowe i testowe
    # 80% danych pójdzie do trenowania, 20% do testowania modelu.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Budowa modelu (sieć neuronowa)
    # Sequential oznacza model warstwa po warstwie.
    model = keras.Sequential([
        # Pierwsza warstwa: 16 neuronów, aktywacja ReLU, wejście = liczba cech z X
        keras.layers.Dense(16, activation="relu", input_shape=(X.shape[1],)),
        # Druga warstwa: 8 neuronów, aktywacja ReLU
        keras.layers.Dense(8, activation="relu"),
        # Ostatnia warstwa: 1 neuron, aktywacja sigmoid (wynik w przedziale 0–1 → prawdopodobieństwo)
        keras.layers.Dense(1, activation="sigmoid")
    ])

    # 5. Kompilacja modelu
    # Wybieramy optymalizator Adam z podanym learning_rate
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",   # funkcja kosztu dla klasyfikacji 0/1
        metrics=["accuracy"]          # chcemy śledzić dokładność
    )

    # 6. Trenowanie modelu
    # model.fit uruchamia proces uczenia – sieć "dopasowuje się" do danych
    history = model.fit(
        X_train, y_train,             # dane treningowe
        validation_data=(X_test, y_test),  # dane walidacyjne (testowe podczas uczenia)
        epochs=epochs,                # ile razy sieć zobaczy wszystkie dane
        batch_size=batch_size,        # ile próbek przetwarzać naraz
        verbose=1                     # poziom logów (1 = pokaż pasek postępu)
    )

    # 7. Wizualizacja wyników
    # Tworzymy jeden obrazek z dwoma podwykresami: loss i accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Wykres strat (loss)
    ax1.plot(history.history['loss'], label='Train Loss')           # strata na treningu
    ax1.plot(history.history['val_loss'], label='Validation Loss')  # strata na walidacji
    ax1.set_xlabel("Epoka")     # oś X = numer epoki
    ax1.set_ylabel("Loss")      # oś Y = strata
    ax1.legend()
    ax1.set_title("Krzywa Loss")

    # Wykres dokładności (accuracy)
    ax2.plot(history.history['accuracy'], label='Train Acc')             # dokładność na treningu
    ax2.plot(history.history['val_accuracy'], label='Validation Acc')    # dokładność na walidacji
    ax2.set_xlabel("Epoka")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.set_title("Krzywa Accuracy")

    # Dostosowanie układu i pokazanie wykresu
    plt.tight_layout()
    plt.show()

    # Zwracamy wytrenowany model, żeby można go było później używać do predykcji
    return model


# Wywołanie funkcji
# Uczymy model przez 30 epok, batch size = 16, learning rate = 0.001
model = train_model(epochs=30, batch_size=16, learning_rate=0.001)
