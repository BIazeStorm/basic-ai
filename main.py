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

st.set_page_config(page_title="Лабораторна 5: Мережа Хопфілда", layout="wide")


# ---------- Функції ----------
def extract_vectors_from_pil(img_pil, grid=GRID_SIZE, background_mode="auto"):
    """Обчислює абсолютний і нормований вектори ознак."""
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
    abs_vec = np.array(abs_vec, dtype=float)
    max_val = np.max(abs_vec) if np.max(abs_vec) != 0 else 1.0
    norm_vec = abs_vec / max_val
    return abs_vec, norm_vec, background_mode


def binarize_vector(vec):
    """Бінаризує вектор до значень +1 та -1."""
    threshold = np.mean(vec)
    # Якщо всі значення однакові, середнє буде дорівнювати цьому значенню.
    # Щоб уникнути повністю (-1) вектора, порівнюємо > замість >=.
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


def format_vector_multiline(vec, values_per_row=GRID_SIZE, is_int=True):
    """Форматує вектор у багаторядковий текст."""
    lines = []
    for row in range(0, len(vec), values_per_row):
        chunk = vec[row:row + values_per_row]
        if is_int:
            lines.append(" ".join(f"{int(x):4d}" for x in chunk))
        else:
            lines.append(" ".join(f"{x:6.2f}" for x in chunk))
    return "\n".join(lines)


def format_matrix_multiline(matrix, values_per_row=GRID_SIZE):
    """Форматує матрицю у багаторядковий текст."""
    lines = []
    for row_vec in matrix:
        lines.append(" ".join(f"{int(x):4d}" for x in row_vec))
    return "\n".join(lines)


# ---------- ІНТЕРФЕЙС ----------
st.title("🧠 Лабораторна 5: Мережа Хопфілда")
st.markdown("Система розпізнавання на базі нейронної мережі асоціативного типу.")

# === Ініціалізація станів ===
if "train_data" not in st.session_state:
    st.session_state.train_data = [[], [], []]
if "etalon_vectors" not in st.session_state:
    st.session_state.etalon_vectors = None
if "hopfield_weights" not in st.session_state:
    st.session_state.hopfield_weights = None
if "unknown" not in st.session_state:
    st.session_state.unknown = None
if "recognition_log" not in st.session_state:
    st.session_state.recognition_log = None

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
        uploaded = st.file_uploader(f"Файли для {class_labels[i]}", accept_multiple_files=True, key=f"upl_{i}")
        if uploaded:
            st.session_state.train_data[i] = []
            for up in uploaded:
                try:
                    img = Image.open(up).convert("RGB")
                    abs_v, _, mode = extract_vectors_from_pil(img, GRID_SIZE, bg_mode)
                    st.session_state.train_data[i].append({
                        "name": up.name, "pil": img, "abs": abs_v, "bin": binarize_vector(abs_v)
                    })
                except Exception as e:
                    st.error(f"Помилка з файлом {up.name}: {e}")
            st.success(f"Завантажено: {len(st.session_state.train_data[i])}")

# === Навчання (обчислення матриці ваг) ===
st.write("---")
st.subheader("2️⃣ Навчання мережі (обчислення матриці ваг)")

if st.button("Обчислити матрицю ваг (Train)"):
    all_samples = [s for class_list in st.session_state.train_data for s in class_list]
    if not all_samples:
        st.error("Будь ласка, завантажте хоча б один зразок.")
    else:
        with st.spinner("Обчислення..."):
            st.session_state.etalon_vectors = all_samples
            num_neurons = len(all_samples[0]['bin'])

            # Створюємо матрицю з еталонів [m x n], де m - кількість образів, n - кількість нейронів
            etalon_matrix = np.array([s['bin'] for s in all_samples])

            # Обчислюємо матрицю ваг за правилом Хебба
            weights = np.dot(etalon_matrix.T, etalon_matrix)

            # Обнуляємо діагональ
            np.fill_diagonal(weights, 0)

            st.session_state.hopfield_weights = weights
            st.success("✅ Матриця ваг успішно обчислена.")

# === Відображення результатів навчання ===
if st.session_state.hopfield_weights is not None:
    st.write("---")
    st.subheader("📊 Результати навчання")

    st.markdown("**Еталонні бінаризовані вектори (+1 / -1):**")
    for i, etalon in enumerate(st.session_state.etalon_vectors):
        with st.expander(f"Образ {i + 1}: {etalon['name']}"):
            col1, col2 = st.columns(2)
            col1.image(etalon['pil'], width=150)
            col2.text(format_vector_multiline(etalon['bin']))

    st.markdown("**Матриця ваг (W):**")
    st.text_area("Матриця", format_matrix_multiline(st.session_state.hopfield_weights), height=300)

# === Розпізнавання невідомого образу ===
st.write("---")
st.subheader("3️⃣ Розпізнавання невідомого образу")

unknown_file = st.file_uploader("Завантажити зображення невідомого образу", type=["png", "jpg", "jpeg"])
if unknown_file:
    try:
        unknown_pil = Image.open(unknown_file).convert("RGB")
        abs_u, _, mode = extract_vectors_from_pil(unknown_pil, GRID_SIZE, bg_mode)
        bin_u = binarize_vector(abs_u)
        st.session_state.unknown = {"name": unknown_file.name, "pil": unknown_pil, "abs": abs_u, "bin": bin_u}

        img_col, vec_col = st.columns(2)
        with img_col:
            st.image(image_with_grid_bytes(st.session_state.unknown['pil']), caption=f"Невідомий: {unknown_file.name}")
        with vec_col:
            st.markdown("**Абсолютний вектор:**")
            st.text(format_vector_multiline(st.session_state.unknown['abs']))
            st.markdown("**Бінаризований вектор (+1 / -1):**")
            st.text(format_vector_multiline(st.session_state.unknown['bin']))
    except Exception as e:
        st.error(f"Помилка при читанні файлу: {e}")
        st.session_state.unknown = None

if st.session_state.unknown:
    max_iter = st.slider("Максимальна кількість ітерацій", 1, 100, 10)
    if st.button("Розпізнати образ (Класифікувати)"):
        if st.session_state.hopfield_weights is None:
            st.error("Спочатку потрібно обчислити матрицю ваг.")
        else:
            W = st.session_state.hopfield_weights
            current_state = st.session_state.unknown['bin'].copy()

            log = f"Крок 0 (вхідний вектор):\n{format_vector_multiline(current_state)}\n\n"

            with st.spinner("Процес розпізнавання..."):
                for i in range(max_iter):
                    # Розрахунок нового стану
                    next_state_continuous = np.dot(W, current_state)
                    next_state = np.sign(next_state_continuous)
                    next_state[next_state == 0] = 1  # Обробка випадку, коли добуток = 0

                    log += f"--- Ітерація {i + 1} ---\n"
                    log += f"Стан до активації (W * Y(t)):\n{format_vector_multiline(next_state_continuous, is_int=False)}\n"
                    log += f"Новий стан Y(t+1):\n{format_vector_multiline(next_state)}\n\n"

                    # Перевірка на стабілізацію
                    if np.array_equal(current_state, next_state):
                        log += ">>> Стан мережі стабілізувався. <<<"
                        current_state = next_state
                        break

                    current_state = next_state
                else:
                    log += ">>> Досягнуто максимальної кількості ітерацій. <<<"

            st.session_state.recognition_log = log

            # Пошук найближчого еталона
            best_match_idx = -1
            max_similarity = -1
            for idx, etalon in enumerate(st.session_state.etalon_vectors):
                similarity = np.sum(current_state == etalon['bin'])
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match_idx = idx

            num_neurons = len(current_state)
            similarity_percent = (max_similarity / num_neurons) * 100
            result_text = f"Образ розпізнано як **{st.session_state.etalon_vectors[best_match_idx]['name']}** (схожість: {similarity_percent:.1f}%)"
            st.session_state.classification_result = result_text

            st.subheader("🏁 Результат класифікації")
            st.markdown(result_text)
            st.image(st.session_state.etalon_vectors[best_match_idx]['pil'], width=200,
                     caption=f"Розпізнаний образ: {st.session_state.etalon_vectors[best_match_idx]['name']}")

            with st.expander("Показати детальний лог розпізнавання"):
                st.text_area("Лог", log, height=400)

# === Збереження звіту ===
st.write("---")
st.subheader("4️⃣ Збереження звіту")

if st.button("Зберегти звіт"):
    if st.session_state.hopfield_weights is None or st.session_state.unknown is None:
        st.error("Потрібно виконати навчання та завантажити невідомий образ.")
    else:
        unknown_name = os.path.splitext(st.session_state.unknown['name'])[0]
        filename = f"report_Hopfield_{unknown_name}.txt"

        with open(filename, "w", encoding="utf-8") as f:
            f.write("=" * 40 + "\n")
            f.write("   ЗВІТ ПРО РОБОТУ МЕРЕЖІ ХОПФІЛДА\n")
            f.write("=" * 40 + "\n\n")

            f.write("--- 1. Еталонні образи ---\n")
            for etalon in st.session_state.etalon_vectors:
                f.write(f"\nОбраз: {etalon['name']}\n")
                f.write("Бінаризований вектор:\n")
                f.write(format_vector_multiline(etalon['bin']) + "\n")

            f.write("\n\n--- 2. Матриця ваг (W) ---\n")
            f.write(format_matrix_multiline(st.session_state.hopfield_weights) + "\n")

            f.write("\n\n--- 3. Невідомий образ ---\n")
            f.write(f"Ім'я файлу: {st.session_state.unknown['name']}\n")
            f.write("Вхідний бінаризований вектор:\n")
            f.write(format_vector_multiline(st.session_state.unknown['bin']) + "\n")

            f.write("\n\n--- 4. Процес розпізнавання ---\n")
            if st.session_state.recognition_log:
                f.write(st.session_state.recognition_log)
            else:
                f.write("Розпізнавання не проводилось.\n")

            f.write("\n\n--- 5. Результат ---\n")
            if st.session_state.classification_result:
                f.write(st.session_state.classification_result.replace("**", "") + "\n")
            else:
                f.write("Результат відсутній.\n")

        st.success(f"✅ Звіт успішно збережено у файл: {filename}")

st.write("---")
st.caption("© Mykola Krasnov")