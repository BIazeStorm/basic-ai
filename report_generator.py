import numpy as np
from utils import format_vector_multiline


def save_report(filename, etalon_vectors_data, hamming_params, unknown_data, recognition_log,
                classification_result_text):
    """
    Зберігає повний звіт у текстовий файл.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=" * 40 + "\n")
        f.write("   ЗВІТ ПРО РОБОТУ МЕРЕЖІ ХЕММІНГА\n")
        f.write("=" * 40 + "\n\n")

        f.write("--- 1. Еталонні образи ---\n")
        if not etalon_vectors_data:
            f.write("Еталони не завантажені.\n")
        for etalon in etalon_vectors_data:
            f.write(f"\nОбраз: {etalon['name']}\n")
            f.write("Бінаризований вектор:\n")
            f.write(format_vector_multiline(etalon['bin']) + "\n")

        f.write("\n\n--- 2. Параметри мережі ---\n")
        if not hamming_params:
            f.write("Параметри не обчислені.\n")
        else:
            f.write(f"Параметр гальмування (ε): {hamming_params['epsilon']:.4f}\n\n")
            f.write("Ваги 1-го шару (W = X/2):\n" + np.array2string(hamming_params['weights_L1'], precision=2) + "\n\n")
            f.write("Пороги 1-го шару (B = n/2):\n" + np.array2string(hamming_params['biases_L1'], precision=2) + "\n")

        f.write("\n\n--- 3. Невідомий образ ---\n")
        if not unknown_data:
            f.write("Невідомий образ не завантажений.\n")
        else:
            f.write(f"Ім'я файлу: {unknown_data['name']}\n")
            f.write("Вхідний бінаризований вектор:\n")
            f.write(format_vector_multiline(unknown_data['bin']) + "\n")

        f.write("\n\n--- 4. Процес розпізнавання (лог) ---\n")
        f.write(recognition_log or "Розпізнавання не проводилось.\n")

        f.write("\n\n--- 5. Результат ---\n")
        f.write(classification_result_text or "Результат відсутній.\n")