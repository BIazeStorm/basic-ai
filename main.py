import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import io
import os
import time

# ---------- Налаштування ----------
GRID_SIZE = 5  # 5x5 сітка
NON_WHITE_THRESHOLD = 240  # поріг для "білий" фон
NON_BLACK_THRESHOLD = 15  # поріг для "чорний" фон

st.set_page_config(page_title="Лабораторна 4: Перцептрон", layout="wide")


# ---------- Функції ----------
def extract_vectors_from_pil(img_pil, grid=GRID_SIZE, background_mode="auto"):
    """
    Обчислює абсолютний і нормований вектори ознак.
    Працює з будь-яким розміром зображення.
    background_mode = 'auto' | 'white' | 'black'
    """
    img = img_pil.convert("RGB")
    arr = np.array(img)
    h, w, _ = arr.shape

    # === Визначення типу фону ===
    if background_mode == "auto":
        mean_brightness = np.mean(arr)
        if mean_brightness > 127:
            background_mode = "white"
        else:
            background_mode = "black"

    # === Умова для визначення "пікселів об'єкта" ===
    if background_mode == "white":
        # об'єкт темний, фон світлий
        object_condition = lambda cell: np.any(cell < NON_WHITE_THRESHOLD, axis=2)
    else:
        # об'єкт світлий, фон темний
        object_condition = lambda cell: np.any(cell > NON_BLACK_THRESHOLD, axis=2)

    # === Поділ на клітинки ===
    abs_vec = []
    cell_height = h // grid
    cell_width = w // grid

    for i in range(grid):
        for j in range(grid):
            y1, y2 = i * cell_height, (i + 1) * cell_height
            x1, x2 = j * cell_width, (j + 1) * cell_width
            cell = arr[y1:y2, x1:x2]
            mask = object_condition(cell)
            abs_vec.append(int(np.sum(mask)))

    abs_vec = np.array(abs_vec, dtype=float)
    max_val = np.max(abs_vec) if np.max(abs_vec) != 0 else 1.0
    norm_vec = abs_vec / max_val

    return abs_vec, norm_vec, background_mode


def image_with_grid_bytes(img_pil, grid=GRID_SIZE):
    """Повертає байти PNG для показу в streamlit з накладеною сіткою."""
    img = img_pil.convert("RGB").copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for i in range(1, grid):
        x = i * w / grid
        y = i * h / grid
        draw.line((x, 0, x, h), fill=(255, 0, 0), width=1)
        draw.line((0, y, w, y), fill=(255, 0, 0), width=1)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)
    return bio


def format_vector_multiline(vec, values_per_row=None):
    """Форматує вектор у багаторядковий текст."""
    if values_per_row is None:
        values_per_row = GRID_SIZE
    lines = []
    vec_int = [int(x) for x in vec]
    for row in range(0, len(vec_int), values_per_row):
        chunk = vec_int[row:row + values_per_row]
        lines.append(" ".join(f"{x:6d}" for x in chunk))
    return "\n".join(lines)


def format_norm_vector_multiline(vec, values_per_row=None):
    """Форматує нормований вектор у багаторядковий текст."""
    if values_per_row is None:
        values_per_row = GRID_SIZE
    lines = []
    for row in range(0, len(vec), values_per_row):
        chunk = vec[row:row + values_per_row]
        lines.append(" ".join(f"{x:.3f}" for x in chunk))
    return "\n".join(lines)


# ---------- ІНТЕРФЕЙС ----------
st.title("🧠 Лабораторна 4: Реалізація системи розпізнавання перцептронного типу")

st.markdown("""
Ця програма виконує побудову ознакових векторів, навчання одношарового перцептрона для розпізнавання двох класів
та класифікацію невідомого зображення.
""")

# === Ініціалізація станів ===
if "train_data" not in st.session_state:
    st.session_state.train_data = [[], []]
if "perceptron_weights" not in st.session_state:
    st.session_state.perceptron_weights = None
if "training_log" not in st.session_state:
    st.session_state.training_log = ""
if "unknown" not in st.session_state:
    st.session_state.unknown = None
if "classification_result" not in st.session_state:
    st.session_state.classification_result = None

# === Налаштування та завантаження ===
st.write("---")
st.subheader("1️⃣ Налаштування та завантаження зразків")

# === Вибір типу фону ===
col_settings_1, col_settings_2 = st.columns(2)
with col_settings_1:
    bg_choice = st.radio("Тип фону зображень:", ["Автоматично", "Білий", "Чорний"], horizontal=True)
    bg_mode_map = {"Автоматично": "auto", "Білий": "white", "Чорний": "black"}
    bg_mode = bg_mode_map[bg_choice]

# === Назви класів ===
col_names = st.columns(2)
class_labels = []
for i, c in enumerate(col_names):
    default_val = f"Клас A (+1)" if i == 0 else f"Клас B (-1)"
    lbl = c.text_input(f"Назва класу {i + 1}", value=default_val)
    class_labels.append(lbl.strip() if lbl.strip() else default_val)

# === Завантаження навчальних зразків ===
cols = st.columns(2)
for i, col in enumerate(cols):
    with col:
        st.markdown(f"**{class_labels[i]}**")
        uploaded = st.file_uploader(f"Файли для {class_labels[i]} (10 зразків)", accept_multiple_files=True,
                                    key=f"upl_{i}")
        if uploaded:
            st.session_state.train_data[i] = []
            for up in uploaded:
                try:
                    img = Image.open(up).convert("RGB")
                except Exception as e:
                    st.error(f"Не вдалося відкрити {up.name}: {e}")
                    continue
                abs_v, norm_v, mode_used = extract_vectors_from_pil(img, GRID_SIZE, bg_mode)
                st.session_state.train_data[i].append({
                    "name": up.name,
                    "pil": img,
                    "abs": abs_v,
                    "norm": norm_v,
                    "mode": mode_used
                })
            st.success(f"Завантажено {len(st.session_state.train_data[i])} файлів для {class_labels[i]}")

# === Навчання перцептрона ===
st.write("---")
st.subheader("2️⃣ Навчання перцептрона")

train_params_col, train_btn_col = st.columns([1, 1])

with train_params_col:
    learning_rate = st.number_input("Швидкість навчання (r)", min_value=0.001, max_value=1.0, value=0.1, step=0.01)
    epochs = st.number_input("Максимальна кількість епох", min_value=1, max_value=10000, value=100, step=10)

with train_btn_col:
    st.write("")
    st.write("")
    if st.button("Навчити перцептрон", use_container_width=True):
        # Перевірка наявності даних
        if not (st.session_state.train_data[0] and st.session_state.train_data[1]):
            st.error("Потрібно завантажити хоча б один зразок у кожен клас.")
        else:
            # Підготовка даних для навчання
            training_set = []
            for sample in st.session_state.train_data[0]:
                training_set.append((sample['norm'], 1))  # Клас A -> +1
            for sample in st.session_state.train_data[1]:
                training_set.append((sample['norm'], -1))  # Клас B -> -1

            # Ініціалізація ваг
            num_features = len(training_set[0][0])
            weights = np.random.uniform(0, 1, num_features)
            st.session_state.training_log = ""

            with st.spinner("Триває навчання..."):
                start_time = time.time()
                for epoch in range(epochs):
                    errors = 0
                    for input_vector, desired_output in training_set:
                        # Обчислення виходу
                        weighted_sum = np.dot(input_vector, weights)
                        output = 1 if weighted_sum >= 0 else -1

                        # Корекція ваг у разі помилки
                        if output != desired_output:
                            error = desired_output - output
                            weights += learning_rate * error * input_vector
                            errors += 1

                    log_entry = f"Епоха {epoch + 1}/{epochs}: помилок = {errors}\n"
                    st.session_state.training_log += log_entry

                    if errors == 0:
                        st.session_state.training_log += "\nНавчання успішно завершено (немає помилок)!\n"
                        break

                end_time = time.time()
                st.session_state.training_log += f"\nЧас навчання: {end_time - start_time:.2f} сек."

            st.session_state.perceptron_weights = weights
            st.success("✅ Навчання завершено.")

# === Відображення результатів навчання ===
if st.session_state.perceptron_weights is not None:
    st.write("---")
    st.subheader("📊 Результати навчання")
    log_col, weights_col = st.columns(2)
    with log_col:
        st.markdown("**Лог навчання:**")
        st.text_area("Лог", st.session_state.training_log, height=250)
    with weights_col:
        st.markdown("**Фінальний вектор ваг (w):**")
        st.text(format_norm_vector_multiline(st.session_state.perceptron_weights))

# === Розпізнавання невідомого образу ===
st.write("---")
st.subheader("3️⃣ Розпізнавання невідомого образу")

unknown_file = st.file_uploader("Завантажити зображення невідомого образу", type=["png", "jpg", "jpeg"],
                                key="unknown_file")
if unknown_file:
    try:
        unknown_pil = Image.open(unknown_file).convert("RGB")
        st.session_state.unknown = {"name": unknown_file.name, "pil": unknown_pil}

        abs_u, norm_u, mode_used = extract_vectors_from_pil(unknown_pil, GRID_SIZE, bg_mode)
        st.session_state.unknown["abs"] = abs_u
        st.session_state.unknown["norm"] = norm_u

        img_col, vec_col = st.columns(2)
        with img_col:
            st.image(image_with_grid_bytes(unknown_pil, GRID_SIZE), caption=f"Невідомий: {unknown_file.name}")
        with vec_col:
            st.markdown(f"**Тип фону (використано):** {mode_used}")
            st.markdown("**Нормований вектор:**")
            st.text(format_norm_vector_multiline(norm_u))

    except Exception as e:
        st.error(f"Помилка при читанні файлу: {e}")
        st.session_state.unknown = None

if st.session_state.unknown:
    if st.button("Класифікувати", use_container_width=True):
        if st.session_state.perceptron_weights is None:
            st.error("Перцептрон ще не навчений. Спочатку натисніть 'Навчити перцептрон'.")
        else:
            unknown_vector = st.session_state.unknown["norm"]
            weights = st.session_state.perceptron_weights

            weighted_sum = np.dot(unknown_vector, weights)
            output = 1 if weighted_sum >= 0 else -1

            if output == 1:
                result_class = class_labels[0]
            else:
                result_class = class_labels[1]

            result_text = f"**Результат:** Зображення належить до **{result_class}** (вихід перцептрона: {output})"
            st.session_state.classification_result = result_text
            st.markdown(f"### {result_text}")

# === Збереження ===
st.write("---")
st.subheader("4️⃣ Збереження результатів у .txt")

if st.button("Зберегти звіт"):
    if st.session_state.perceptron_weights is None or st.session_state.unknown is None:
        st.error("Потрібно спочатку виконати навчання та завантажити невідомий образ.")
    else:
        # Формування імені файлу
        parts = []
        for i in range(2):
            if st.session_state.train_data[i]:
                parts.append(os.path.splitext(st.session_state.train_data[i][0]["name"])[0])
            else:
                parts.append(f"class_{i + 1}")
        unknown_name = os.path.splitext(st.session_state.unknown["name"])[0]
        filename = f"report_{parts[0]}_vs_{parts[1]}_on_{unknown_name}.txt".replace(" ", "_")

        with open(filename, "w", encoding="utf-8") as f:
            f.write("=== Звіт про роботу перцептрона ===\n\n")
            f.write(f"Дата та час: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("--- Параметри навчання ---\n")
            f.write(f"Швидкість навчання: {learning_rate}\n")
            f.write(f"Макс. кількість епох: {epochs}\n\n")

            f.write("--- Навчальні дані ---\n")
            for i in range(2):
                f.write(f"\nКлас {i + 1}: {class_labels[i]}\n")
                f.write(f"Кількість зразків: {len(st.session_state.train_data[i])}\n")
                for j, s in enumerate(st.session_state.train_data[i]):
                    f.write(f"  Зразок {j + 1}: {s['name']}\n")
                    f.write(f"    Нормований вектор: {np.round(s['norm'], 3).tolist()}\n")

            f.write("\n--- Результати навчання ---\n")
            f.write("Лог навчання:\n")
            f.write(st.session_state.training_log + "\n")
            f.write("Фінальний вектор ваг:\n")
            f.write(str(np.round(st.session_state.perceptron_weights, 4).tolist()))
            f.write("\n\n")

            f.write("--- Розпізнавання невідомого образу ---\n")
            f.write(f"Ім'я файлу: {st.session_state.unknown['name']}\n")
            f.write(f"Нормований вектор: {np.round(st.session_state.unknown['norm'], 3).tolist()}\n\n")

            f.write("--- Результат класифікації ---\n")
            if st.session_state.classification_result:
                f.write(st.session_state.classification_result.replace("**", "") + "\n")
            else:
                f.write("Класифікація не проводилась.\n")

        st.success(f"Результати збережено у файл: {filename}")

st.write("---")
st.caption("© Mykola Krasnov")