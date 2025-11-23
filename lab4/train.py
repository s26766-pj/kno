import argparse
import os
import pandas as pd
import keras_tuner as kt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical

import joblib


# ============================================================
# 1. Parsowanie argumentów CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train wine classifier with Keras Tuner")

    parser.add_argument("--max_trials", type=int, default=10,
                        help="Ile konfiguracji hiperparametrów testować")
    parser.add_argument("--executions", type=int, default=1,
                        help="Ile razy trenować każdą konfigurację")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Maksymalna liczba epok")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size (gdy używany przez tunera)")
    parser.add_argument("--tuner", type=str, choices=["random", "bayesian", "hyperband"],
                        default="random", help="Rodzaj tunera Keras")

    return parser.parse_args()


# ============================================================
# 2. Wczytanie danych
# ============================================================

def load_data():
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

    return X_train, X_test, y_train, y_test, scaler


# ============================================================
# 3. Funkcja budująca model (KERAS TUNER)
# ============================================================

def build_model(hp: kt.HyperParameters):
    model = Sequential(name="WineModel")

    # Ilość neuronów w 1 warstwie
    hp_units1 = hp.Int("units_layer1", min_value=8, max_value=128, step=8)
    model.add(Dense(hp_units1, activation="relu", input_shape=(13,)))

    # Batch Norm jako opcjonalna
    if hp.Boolean("use_batchnorm_1"):
        model.add(BatchNormalization())

    # Dropout w 1 warstwie
    hp_dropout1 = hp.Float("dropout_1", 0.0, 0.5, step=0.1)
    model.add(Dropout(hp_dropout1))

    # Druga warstwa (opcjonalna)
    if hp.Boolean("use_second_layer"):
        units2 = hp.Int("units_layer2", min_value=8, max_value=64, step=8)
        model.add(Dense(units2, activation="relu"))
        if hp.Boolean("use_batchnorm_2"):
            model.add(BatchNormalization())
        dropout2 = hp.Float("dropout_2", 0.0, 0.5, step=0.1)
        model.add(Dropout(dropout2))

    # Wyjście
    model.add(Dense(3, activation="softmax"))

    # Learning rate tunowany
    lr = hp.Float("learning_rate", min_value=1e-4, max_value=1e-1, sampling="log")
    optimizer = Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=CategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 4. Wybór tunera
# ============================================================

def get_tuner(tuner_name, build_fn, args, X_train, y_train):
    if tuner_name == "random":
        return kt.RandomSearch(
            build_fn,
            objective="val_accuracy",
            max_trials=args.max_trials,
            executions_per_trial=args.executions,
            directory="tuner_results",
            project_name="random_tuner"
        )

    elif tuner_name == "bayesian":
        return kt.BayesianOptimization(
            build_fn,
            objective="val_accuracy",
            max_trials=args.max_trials,
            executions_per_trial=args.executions,
            directory="tuner_results",
            project_name="bayesian_tuner"
        )

    elif tuner_name == "hyperband":
        return kt.Hyperband(
            build_fn,
            objective="val_accuracy",
            max_epochs=args.epochs,
            directory="tuner_results",
            project_name="hyperband_tuner"
        )


# ============================================================
# 5. Główna funkcja
# ============================================================

def main():
    args = parse_args()

    X_train, X_test, y_train, y_test, scaler = load_data()

    tuner = get_tuner(args.tuner, build_model, args, X_train, y_train)

    tuner.search(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # Najlepszy model
    best_model = tuner.get_best_models(num_models=1)[0]

    # Zapis
    os.makedirs("models", exist_ok=True)
    best_model.save("models/best_wine_model.h5")
    joblib.dump(scaler, "models/scaler.save")

    print("\n===============================")
    print("Najlepsze hiperparametry:")
    best_hp = tuner.get_best_hyperparameters()[0]
    print(best_hp.values)
    print("===============================")


if __name__ == "__main__":
    main()
