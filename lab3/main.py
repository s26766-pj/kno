import argparse

import joblib
import numpy as np
from tensorflow.keras.models import load_model

def parse_arguments():
    parser = argparse.ArgumentParser(description="Wine Classification using Softmax Neural Network")

    parser.add_argument("--model", type=str, default="models/modelB.h5",
                        help="Ścieżka do wytrenowanego modelu (.h5). Domyślnie modelB.h5")

    parser.add_argument("--alcohol", type=float, required=True)
    parser.add_argument("--malic_acid", type=float, required=True)
    parser.add_argument("--ash", type=float, required=True)
    parser.add_argument("--alcalinity", type=float, required=True)
    parser.add_argument("--magnesium", type=float, required=True)
    parser.add_argument("--total_phenols", type=float, required=True)
    parser.add_argument("--flavanoids", type=float, required=True)
    parser.add_argument("--nonflavanoid_phenols", type=float, required=True)
    parser.add_argument("--proanthocyanins", type=float, required=True)
    parser.add_argument("--color_intensity", type=float, required=True)
    parser.add_argument("--hue", type=float, required=True)
    parser.add_argument("--od280_od315", type=float, required=True)
    parser.add_argument("--proline", type=float, required=True)

    return parser.parse_args()

def main():
    args = parse_arguments()

    # Załaduj model (trening z wykorzystaniem categorical_crossentropy + softmax)
    model = load_model(args.model)

    # Wektor wejściowy dla predykcji – musi mieć 13 cech
    user_input = np.array([[args.alcohol,
                   args.malic_acid,
                   args.ash,
                   args.alcalinity,
                   args.magnesium,
                   args.total_phenols,
                   args.flavanoids,
                   args.nonflavanoid_phenols,
                   args.proanthocyanins,
                   args.color_intensity,
                   args.hue,
                   args.od280_od315,
                   args.proline]])

    scaler = joblib.load("models/scaler.save")
    user_input_scaled = scaler.transform(user_input)

    # Predykcja modelu Softmax (zwraca prawdopodobieństwa klas)
    pred = model.predict(user_input_scaled)
    predicted_class = np.argmax(pred, axis=1)[0] + 1  # +1 bo klasy to 1,2,3

    print("\n===== WYNIK KLASYFIKACJI WINA =====")
    print(f"Przewidywana klasa wina: **{predicted_class}**")

    print("\nPrawdopodobieństwa dla każdej klasy:")
    for i, p in enumerate(pred[0]):
        print(f"  Klasa {i+1}: {p:.4f}")

    print("\n(Uwaga: Model został wytrenowany z użyciem Categorical Cross-Entropy – idealny dla klasyfikacji wieloklasowej.)")


if __name__ == "__main__":
    main()
