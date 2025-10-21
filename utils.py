import numpy as np
from PIL import Image, ImageDraw
import io

# --- Налаштування ---
GRID_SIZE = 5
NON_WHITE_THRESHOLD = 240
NON_BLACK_THRESHOLD = 15

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
            cell = arr[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
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