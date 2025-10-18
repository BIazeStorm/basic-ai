import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import io
import os
import time

# ---------- Налаштування ----------
GRID_SIZE = 5  # 5x5 сітка
NON_WHITE_THRESHOLD = 240  # Поріг для "білого" фону
NON_BLACK_THRESHOLD = 15  # Поріг для "чорного" фону

st.set_page_config(page_title="Лабораторна 6: Мережа Хеммінга", layout="wide")


# ---------- Функції ----------
def extract_vectors_from_pil(img_pil, grid=GRID_SIZE, background_mode="auto"):
    """Обчислює абсолютний вектор ознак."""
    img = img_pil.convert("RGB")
    arr = np.array(img)
    h, w, _ = arr.shape
    if background_mode == "auto":
        background_mode = "white" if np.mean(arr) > 127 else "black"
    if background_mode == "white":
        object_condition = lambda cell: np.any(cell < NON_WHITE_THRESHOLD, axis=2)
    else:
        object_condition = lambda cell: np.any(cell > NON_BLACK_THRESHOLD, axis=2)
    abs_vec = []
    cell_height, cell_width = h // grid, w // grid
    for i in range(grid):
        for j in range(grid):
            cell = arr[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]
            abs_vec.append(int(np.sum(object_condition(cell))))
    return np.array(abs_vec, dtype=float), background_mode


def binarize_vector(vec):
    """Бінаризує вектор до значень +1 та -1."""
    threshold = np.mean(vec)
    return np.where(vec > threshold, 1, -1)


def image_with_grid_bytes(img_pil, grid=GRID_SIZE):
    """Повертає байти PNG з накладеною сіткою."""
    img = img_pil.convert("RGB").copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for i in range(1, grid):
        x, y = i * w / grid, i * h / grid
        draw.line((x, 0, x, h), fill=(255, 0, 0), width=1)
        draw.line((0, y, w, y), fill=(255, 0, 0), width=1)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


def format_vector_multiline(vec, values_per_row=GRID_SIZE, precision=0):
    """Форматує вектор у багаторядковий текст."""
    lines = []
    for row in range(0, len(vec), values_per_row):
        chunk = vec[row:row + values_per_row]
        if precision == 0:
            lines.append(" ".join(f"{int(x):4d}" for x in chunk))
        else:
            lines.append(" ".join(f"{x:6.{precision}f}" for x in chunk))
    return "\n".join(lines)


# ---------- ІНТЕРФЕЙС ----------
st.title("🧠 Лабораторна 6: Мережа Хеммінга")
st.markdown("Система розпізнавання, що реалізує оптимальний класифікатор мінімуму похибки.")

# === Ініціалізація станів ===
if "train_data" not in st.session_state:
    st.session_state.train_data = [[], [], []]
if "etalon_vectors" not in st.session_state:
    st.session_state.etalon_vectors = None
if "hamming_params" not in st.session_state:
    st.session_state.hamming_params = None
if "unknown" not in st.session_state:
    st.session_state.unknown = None
if "recognition_log" not in st.session_state:
    st.session_state.recognition_log = None
if "classification_result" not in st.session_state:
    st.session_state.classification_result = None

# === Завантаження зразків ===
st.write("---")
st.subheader("1️⃣ Завантаження зразкових зображень (еталонів)")

col_settings, _ = st.columns(2)
with col_settings:
    bg_choice = st.radio("Тип фону зображень:", ["Автоматично", "Білий", "Чорний"], horizontal=True)
    bg_mode = {"Автоматично": "auto", "Білий": "white", "Чорний": "black"}[bg_choice]

col_names = st.columns(3)
class_labels = [c.text_input(f"Назва класу {i + 1}", value=f"Клас {i + 1}") for i, c in enumerate(col_names)]

cols = st.columns(3)
for i, col in enumerate(cols):
    with col:
        st.markdown(f"**{class_labels[i]}**")
        uploaded = st.file_uploader(f"Файл для {class_labels[i]} (один еталон)", key=f"upl_{i}")
        if uploaded:
            try:
                img = Image.open(uploaded).convert("RGB")
                abs_v, _ = extract_vectors_from_pil(img, GRID_SIZE, bg_mode)
                st.session_state.train_data[i] = [{
                    "name": uploaded.name, "pil": img, "abs": abs_v, "bin": binarize_vector(abs_v)
                }]
                st.success(f"Завантажено: {uploaded.name}")
            except Exception as e:
                st.error(f"Помилка з файлом {uploaded.name}: {e}")

# === Навчання (обчислення параметрів мережі) ===
st.write("---")
st.subheader("2️⃣ Навчання мережі (обчислення параметрів)")

if st.button("Обчислити параметри мережі (Train)"):
    # Збираємо лише один еталон з кожного завантаженого класу
    all_samples = [class_list[0] for class_list in st.session_state.train_data if class_list]

    if len(all_samples) < 2:
        st.error("Будь ласка, завантажте еталони хоча б для двох класів.")
    else:
        with st.spinner("Обчислення..."):
            st.session_state.etalon_vectors = all_samples
            num_neurons = len(all_samples[0]['bin'])  # n
            num_etalons = len(all_samples)  # m

            # Формуємо матрицю еталонів [m x n]
            etalon_matrix = np.array([s['bin'] for s in all_samples])

            # Ініціалізація першого шару
            weights_L1 = etalon_matrix.T / 2  # [n x m]
            biases_L1 = np.full(num_etalons, num_neurons / 2)

            # Ініціалізація другого шару (MAXNET)
            epsilon = 1 / (num_etalons + 1)  # Значення, що задовольняє умову 0 < ε < 1/m

            st.session_state.hamming_params = {
                "weights_L1": weights_L1,
                "biases_L1": biases_L1,
                "epsilon": epsilon
            }
            st.success("✅ Параметри мережі успішно обчислені.")

# === Відображення результатів навчання ===
if st.session_state.hamming_params:
    st.write("---")
    st.subheader("📊 Параметри навченої мережі")

    st.markdown("**Еталонні бінаризовані вектори (+1 / -1):**")
    for etalon in st.session_state.etalon_vectors:
        with st.expander(f"Еталон: {etalon['name']}"):
            col1, col2 = st.columns(2)
            col1.image(etalon['pil'], width=150)
            col2.text(format_vector_multiline(etalon['bin']))

    st.markdown(f"**Параметр гальмування (ε) для 2-го шару:** `{st.session_state.hamming_params['epsilon']:.4f}`")

    with st.expander("Показати ваги та пороги 1-го шару"):
        st.markdown("**Ваги 1-го шару (W = X/2):**")
        st.text(st.session_state.hamming_params['weights_L1'])
        st.markdown("**Пороги 1-го шару (B = n/2):**")
        st.text(st.session_state.hamming_params['biases_L1'])

# === Розпізнавання невідомого образу ===
st.write("---")
st.subheader("3️⃣ Розпізнавання невідомого образу")

unknown_file = st.file_uploader("Завантажити зображення невідомого образу", type=["png", "jpg", "jpeg"])
if unknown_file:
    try:
        abs_u, _ = extract_vectors_from_pil(Image.open(unknown_file), GRID_SIZE, bg_mode)
        st.session_state.unknown = {
            "name": unknown_file.name,
            "pil": Image.open(unknown_file).convert("RGB"),
            "abs": abs_u,
            "bin": binarize_vector(abs_u)
        }
        img_col, vec_col = st.columns(2)
        with img_col:
            st.image(image_with_grid_bytes(st.session_state.unknown['pil']), caption=f"Невідомий: {unknown_file.name}")
        with vec_col:
            st.markdown("**Бінаризований вектор (+1 / -1):**")
            st.text(format_vector_multiline(st.session_state.unknown['bin']))
    except Exception as e:
        st.error(f"Помилка при читанні файлу: {e}")
        st.session_state.unknown = None

if st.session_state.unknown:
    max_iter = st.slider("Макс. ітерацій для 2-го шару (MAXNET)", 1, 100, 20)
    if st.button("Розпізнати образ (Класифікувати)"):
        if not st.session_state.hamming_params:
            st.error("Спочатку потрібно обчислити параметри мережі.")
        else:
            params = st.session_state.hamming_params
            unknown_vec = st.session_state.unknown['bin']
            log = ""

            with st.spinner("Процес розпізнавання..."):
                # Крок 1: Обчислення виходів першого шару
                y_L1 = np.dot(unknown_vec, params['weights_L1']) + params['biases_L1']
                log += f"Початкові виходи 1-го шару (ініціалізація 2-го):\n{format_vector_multiline(y_L1, precision=2)}\n\n"

                # Крок 2: Ітерації в другому шарі (MAXNET)
                y_L2 = y_L1.copy()
                for i in range(max_iter):
                    y_L2_prev = y_L2.copy()

                    # Оновлюємо виходи нейронів
                    total_activation = np.sum(y_L2)
                    y_L2 = y_L2_prev - params['epsilon'] * (total_activation - y_L2_prev)

                    # Застосовуємо передатну функцію (залишаємо тільки позитивні)
                    y_L2[y_L2 < 0] = 0

                    log += f"Ітерація {i + 1}:\n{format_vector_multiline(y_L2, precision=3)}\n"

                    # Перевірка на збіжність (тільки один переможець або стабілізація)
                    if np.sum(y_L2 > 0) <= 1 or np.array_equal(y_L2, y_L2_prev):
                        log += "\n>>> Мережа стабілізувалася. <<<"
                        break
                else:
                    log += "\n>>> Досягнуто максимальної кількості ітерацій. <<<"

            st.session_state.recognition_log = log

            # Визначення переможця
            winner_index = np.argmax(y_L2) if np.any(y_L2 > 0) else -1

            if winner_index != -1:
                winner_etalon = st.session_state.etalon_vectors[winner_index]
                result_text = f"Образ розпізнано як **{winner_etalon['name']}** (переміг нейрон #{winner_index + 1})."
                st.session_state.classification_result = (result_text, winner_etalon)
            else:
                result_text = "Не вдалося розпізнати образ (всі нейрони згасли)."
                st.session_state.classification_result = (result_text, None)

            st.subheader("🏁 Результат класифікації")
            st.markdown(result_text)
            if winner_index != -1:
                st.image(st.session_state.classification_result[1]['pil'], width=200, caption=f"Розпізнаний еталон")

            with st.expander("Показати детальний лог розпізнавання"):
                st.text_area("Лог", log, height=300)

# === Збереження звіту ===
st.write("---")
st.subheader("4️⃣ Збереження звіту")

if st.button("Зберегти звіт"):
    if not st.session_state.hamming_params or not st.session_state.unknown:
        st.error("Потрібно виконати навчання та завантажити невідомий образ.")
    else:
        unknown_name = os.path.splitext(st.session_state.unknown['name'])[0]
        filename = f"report_Hamming_{unknown_name}.txt"

        with open(filename, "w", encoding="utf-8") as f:
            f.write("=" * 40 + "\n")
            f.write("   ЗВІТ ПРО РОБОТУ МЕРЕЖІ ХЕММІНГА\n")
            f.write("=" * 40 + "\n\n")

            f.write("--- 1. Еталонні образи ---\n")
            for etalon in st.session_state.etalon_vectors:
                f.write(f"\nОбраз: {etalon['name']}\n")
                f.write("Бінаризований вектор:\n")
                f.write(format_vector_multiline(etalon['bin']) + "\n")

            f.write("\n\n--- 2. Параметри мережі ---\n")
            params = st.session_state.hamming_params
            f.write(f"Параметр гальмування (ε): {params['epsilon']:.4f}\n\n")
            f.write("Ваги 1-го шару (W = X/2):\n" + np.array2string(params['weights_L1'], precision=2) + "\n\n")
            f.write("Пороги 1-го шару (B = n/2):\n" + np.array2string(params['biases_L1'], precision=2) + "\n")

            f.write("\n\n--- 3. Невідомий образ ---\n")
            f.write(f"Ім'я файлу: {st.session_state.unknown['name']}\n")
            f.write("Вхідний бінаризований вектор:\n")
            f.write(format_vector_multiline(st.session_state.unknown['bin']) + "\n")

            f.write("\n\n--- 4. Процес розпізнавання (лог) ---\n")
            f.write(st.session_state.recognition_log or "Розпізнавання не проводилось.\n")

            f.write("\n\n--- 5. Результат ---\n")
            if st.session_state.classification_result:
                f.write(st.session_state.classification_result[0].replace("**", "") + "\n")
            else:
                f.write("Результат відсутній.\n")

        st.success(f"✅ Звіт успішно збережено у файл: {filename}")

st.write("---")
st.caption("© Mykola Krasnov")
