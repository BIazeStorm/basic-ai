import os

import streamlit as st
from PIL import Image

import utils
from hamming_network import HammingNetwork
from report_generator import save_report

st.set_page_config(page_title="Лабораторна 6: Мережа Хеммінга", layout="wide")
st.title("Лабораторна 6: Мережа Хеммінга")

# === Ініціалізація станів ===
if "network" not in st.session_state:
    st.session_state.network = HammingNetwork()

if "train_data" not in st.session_state:
    st.session_state.train_data = [[], [], []]
if "etalon_vectors_data" not in st.session_state:
    st.session_state.etalon_vectors_data = None
if "hamming_params" not in st.session_state:
    st.session_state.hamming_params = None
if "unknown" not in st.session_state:
    st.session_state.unknown = None
if "recognition_log" not in st.session_state:
    st.session_state.recognition_log = None
if "classification_result" not in st.session_state:
    st.session_state.classification_result = (None, None)  # (text, etalon_data)

# === 1. Завантаження зразків ===
st.write("---")
st.subheader("Завантаження зразкових зображень (еталонів)")

col_settings, _ = st.columns(2)
with col_settings:
    bg_choice = st.radio("Тип фону:", ["Автоматично", "Білий", "Чорний"], horizontal=True)
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
                abs_v, _ = utils.extract_vectors_from_pil(img, utils.GRID_SIZE, bg_mode)
                st.session_state.train_data[i] = [{
                    "name": uploaded.name, "pil": img, "abs": abs_v, "bin": utils.binarize_vector(abs_v)
                }]
                st.success(f"Завантажено: {uploaded.name}")
            except Exception as e:
                st.error(f"Помилка з файлом {uploaded.name}: {e}")

# === 2. Навчання ===
st.write("---")
st.subheader("Навчання мережі (обчислення параметрів)")

params = {}
if st.button("Обчислити параметри мережі (Train)"):
    all_samples_data = [class_list[0] for class_list in st.session_state.train_data if class_list]

    if len(all_samples_data) < 2:
        st.error("Будь ласка, завантажте еталони хоча б для двох класів.")
    else:
        st.session_state.etalon_vectors_data = all_samples_data
        # Збираємо лише бінарні вектори для навчання
        etalon_bin_vectors = [s['bin'] for s in all_samples_data]

        with st.spinner("Обчислення..."):
            # Викликаємо метод train нашого об'єкта мережі
            params = st.session_state.network.train(etalon_bin_vectors)
            st.session_state.hamming_params = params
            st.success("✅ Параметри мережі успішно обчислені.")

# === 3. Відображення параметрів ===
if st.session_state.hamming_params:
    st.write("---")
    st.subheader("Параметри навченої мережі")

    for etalon in st.session_state.etalon_vectors_data:
        with st.expander(f"Еталон: {etalon['name']}"):
            col1, col2 = st.columns(2)
            col1.image(etalon['pil'], width=150)
            col2.text(utils.format_vector_multiline(etalon['bin']))

    st.markdown(f"**ε:** `{st.session_state.hamming_params['epsilon']:.4f}`")

    with st.expander("Показати ваги та пороги 1-го шару"):
        st.markdown("**Ваги 1-го шару (W = X/2):**")
        st.text(st.session_state.hamming_params['weights_L1'])
        st.markdown("**Пороги 1-го шару (B = n/2):**")
        st.text(st.session_state.hamming_params['biases_L1'])

# === 4. Розпізнавання ===
st.write("---")
st.subheader("Розпізнавання невідомого образу")

unknown_file = st.file_uploader("Завантажити зображення невідомого образу", type=["png", "jpg", "jpeg"])
if unknown_file:
    try:
        img_pil = Image.open(unknown_file).convert("RGB")
        abs_u, _ = utils.extract_vectors_from_pil(img_pil, utils.GRID_SIZE, bg_mode)
        st.session_state.unknown = {
            "name": unknown_file.name, "pil": img_pil, "abs": abs_u, "bin": utils.binarize_vector(abs_u)
        }
        img_col, vec_col = st.columns(2)
        with img_col:
            st.image(utils.image_with_grid_bytes(st.session_state.unknown['pil']),
                     caption=f"Невідомий: {unknown_file.name}")
        with vec_col:
            st.markdown("**Бінаризований вектор (+1 / -1):**")
            st.text(utils.format_vector_multiline(st.session_state.unknown['bin']))
    except Exception as e:
        st.error(f"Помилка при читанні файлу: {e}")
        st.session_state.unknown = None

if st.session_state.unknown:
    max_iter = st.slider("Макс. ітерацій для 2-го шару (MAXNET)", 1, 100, 20)
    if st.button("Розпізнати образ (Класифікувати)"):
        if not st.session_state.hamming_params:
            st.error("Спочатку потрібно обчислити параметри мережі.")
        else:
            with st.spinner("Процес розпізнавання..."):
                # Викликаємо метод recognize нашого об'єкта мережі
                winner_index, log = st.session_state.network.recognize(
                    st.session_state.unknown['bin'],
                    max_iter
                )

            st.session_state.recognition_log = log

            if winner_index != -1:
                winner_etalon = st.session_state.etalon_vectors_data[winner_index]
                result_text = f"Образ розпізнано як **{winner_etalon['name']}** (переміг нейрон #{winner_index + 1})."
                st.session_state.classification_result = (result_text, winner_etalon)
            else:
                result_text = "Не вдалося розпізнати образ (всі нейрони згасли)."
                st.session_state.classification_result = (result_text, None)

            st.subheader("Результат класифікації")
            st.markdown(result_text)
            if winner_index != -1:
                st.image(st.session_state.classification_result[1]['pil'], width=200, caption=f"Розпізнаний еталон")

            with st.expander("Показати детальний лог розпізнавання"):
                st.text_area("Лог", log, height=300)

# === 5. Збереження звіту ===
st.write("---")
st.subheader("Збереження звіту")

if st.button("Зберегти звіт"):
    if not st.session_state.hamming_params or not st.session_state.unknown:
        st.error("Потрібно виконати навчання та завантажити невідомий образ.")
    else:
        try:
            REPORTS_DIR = "reports"

            if not os.path.exists(REPORTS_DIR):
                os.makedirs(REPORTS_DIR)
                st.toast(f"Створено нову папку: {REPORTS_DIR}")

            unknown_name = os.path.splitext(st.session_state.unknown['name'])[0]
            filename = f"report_Hamming_{unknown_name}.txt"

            full_path = os.path.join(REPORTS_DIR, filename)

            save_report(
                filename=full_path,
                etalon_vectors_data=st.session_state.etalon_vectors_data,
                hamming_params=st.session_state.hamming_params,
                unknown_data=st.session_state.unknown,
                recognition_log=st.session_state.recognition_log,
                classification_result_text=st.session_state.classification_result[0]
            )

            st.success(f"✅ Звіт успішно збережено у файл: {full_path}")

        except Exception as e:
            st.error(f"Помилка при збереженні звіту: {e}")