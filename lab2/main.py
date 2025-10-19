import argparse
import math
import sys
import numpy as np
import tensorflow as tf


# ================================================================
# ZADANIE 1 i 2
# Przygotuj funkcję w TensorFlow, która będzie liczyła obrót
# punktu (x, y) względem punktu (0,0) o zadany kąt (w stopniach).
# ================================================================
@tf.function
def rotate_point_tensor(x, y, angle_degrees):
    # Przeliczenie kąta ze stopni na radiany
    angle_rad = tf.constant(angle_degrees * math.pi / 180, dtype=tf.float32)

    # Tworzymy macierz obrotu 2x2
    rotation_matrix = tf.stack(
        [
            [tf.cos(angle_rad), -tf.sin(angle_rad)],
            [tf.sin(angle_rad), tf.cos(angle_rad)],
        ]
    )

    # Tworzymy tensor reprezentujący punkt
    point = tf.constant([[x], [y]], dtype=tf.float32)

    # Mnożymy macierz obrotu przez punkt (mnożenie macierzowe)
    rotated_point = tf.linalg.matmul(rotation_matrix, point)

    # Zwracamy wynik jako 2-elementowy tensor (x', y')
    return tf.reshape(rotated_point, shape=(2,))

def handle_rotation(args):
    result = rotate_point_tensor(args.x, args.y, args.angle)
    print(f"Rotated point: ({result[0].numpy():.4f}, {result[1].numpy():.4f})")

# ================================================================
# ZADANIE 3
# Przygotuj funkcję, która rozwiązuje układ równań liniowych A*x=b
# metodą macierzową, używając TensorFlow (tf.linalg.solve).
# ================================================================
def solve_linear_system(matrix, vector):
    # Zamieniamy dane wejściowe na tensory
    a = tf.constant(matrix, dtype=tf.float32)
    b = tf.constant(vector, dtype=tf.float32)

    # Rozwiązujemy układ równań liniowych metodą macierzową
    return tf.linalg.solve(a, tf.reshape(b, (-1, 1)))

# ================================================================
# ZADANIE 4 (część 3)
# Funkcja obsługująca rozwiązanie układu równań (komenda "solve")
# ================================================================
def handle_solve(args):
    values = args.values
    total_len = len(values)

    # Automatyczne obliczenie wymiaru macierzy (n)
    # Z równania: liczba parametrów = n^2 + n
    n = int((-1 + math.sqrt(1 + 4 * total_len)) / 2)

    # Sprawdzenie poprawności liczby danych
    if n * n + n != total_len:
        print("Error: Invalid number of parameters for a solvable system.")
        sys.exit(1)

    # Rozdzielenie listy wartości na macierz A i wektor b
    matrix = np.array(values[: n * n], dtype=np.float32).reshape((n, n))
    vector = np.array(values[n * n:], dtype=np.float32)

    # Próba rozwiązania układu przy pomocy TensorFlow
    try:
        result = solve_linear_system(matrix, vector)
        print("Solution:", result.numpy().flatten())
    except tf.errors.InvalidArgumentError as e:
        print("Error: Cannot solve the system. Matrix might be singular.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="TensorFlow math utilities")
    subparsers = parser.add_subparsers(dest="command")

    # ----- opcja 'rotate' -----
    rotate_parser = subparsers.add_parser("rotate", help="Rotate a point")
    rotate_parser.add_argument("x", type=float, help="X coordinate of the point")
    rotate_parser.add_argument("y", type=float, help="Y coordinate of the point")
    rotate_parser.add_argument("angle", type=float, help="Angle in degrees")

    # ----- opcja 'solve' -----
    solve_parser = subparsers.add_parser("solve", help="Solve linear equations")
    solve_parser.add_argument(
        "values", type=float, nargs="+", help="Matrix (n*n) followed by vector (n)"
    )
    return parser.parse_args()

def main():
    args = parse_arguments()

    match args.command:
        case "rotate":
            handle_rotation(args)
        case "solve":
            handle_solve(args)
        case _:
            print("No command specified. Use -h for help.")

if __name__ == "__main__":
    main()
