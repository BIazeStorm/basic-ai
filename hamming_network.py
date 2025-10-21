import numpy as np
from utils import format_vector_multiline


class HammingNetwork:
    def __init__(self):
        self.weights_L1 = None
        self.biases_L1 = None
        self.epsilon = None
        self.num_etalons = 0
        self.num_neurons = 0

    def train(self, etalon_vectors):
        """
        Обчислює та зберігає параметри мережі.
        etalon_vectors - це список (list) бінаризованих numpy-векторів.
        """
        self.num_etalons = len(etalon_vectors)  # m
        self.num_neurons = len(etalon_vectors[0])  # n

        # Формуємо матрицю еталонів [m x n]
        etalon_matrix = np.array(etalon_vectors)

        # Ініціалізація першого шару
        self.weights_L1 = etalon_matrix.T / 2  # [n x m]
        self.biases_L1 = np.full(self.num_etalons, self.num_neurons / 2)

        # Ініціалізація другого шару (MAXNET)
        self.epsilon = 1 / (self.num_etalons + 1)  # 0 < ε < 1/m

        # Повертаємо параметри для відображення в UI
        return {
            "weights_L1": self.weights_L1,
            "biases_L1": self.biases_L1,
            "epsilon": self.epsilon
        }

    def recognize(self, unknown_vec, max_iter=20):
        """
        Розпізнає невідомий вектор.
        Повертає індекс переможця та лог процесу.
        """
        if self.weights_L1 is None:
            raise ValueError("Мережа ще не навчена. Викличте .train() спочатку.")

        log = ""

        # Крок 1: Обчислення виходів першого шару
        y_L1 = np.dot(unknown_vec, self.weights_L1) + self.biases_L1
        log += f"Початкові виходи 1-го шару (ініціалізація 2-го):\n{format_vector_multiline(y_L1, precision=2)}\n\n"

        # Крок 2: Ітерації в другому шарі (MAXNET)
        y_L2 = y_L1.copy()
        for i in range(max_iter):
            y_L2_prev = y_L2.copy()

            # Оновлюємо виходи нейронів
            total_activation = np.sum(y_L2)
            y_L2 = y_L2_prev - self.epsilon * (total_activation - y_L2_prev)

            # Застосовуємо передатну функцію (залишаємо тільки позитивні)
            y_L2[y_L2 < 0] = 0

            log += f"Ітерація {i + 1}:\n{format_vector_multiline(y_L2, precision=3)}\n"

            # Перевірка на збіжність
            active_neurons = np.sum(y_L2 > 0)
            if active_neurons <= 1 and np.array_equal(y_L2, y_L2_prev):
                log += "\n>>> Мережа стабілізувалася (один переможець). <<<"
                break
            if active_neurons == 0:
                log += "\n>>> Мережа стабілізувалася (всі нейрони згасли). <<<"
                break
        else:
            log += "\n>>> Досягнуто максимальної кількості ітерацій. <<<"

        # Визначення переможця
        winner_index = np.argmax(y_L2) if np.any(y_L2 > 0) else -1
        return winner_index, log