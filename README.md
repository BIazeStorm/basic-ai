# 🤖 Basic AI: Classic Neural Network Implementations

This repository is a personal project exploring the foundational algorithms of Artificial Intelligence. It contains from-scratch Python implementations of several classic neural network architectures used for pattern recognition and associative memory.

## 🚀 Implemented Models

This project explores three fundamental neural network architectures.

---

### 1. Rosenblatt's Perceptron
**(Branch: `perceptron`)**

A from-scratch implementation of the single-layer Perceptron, the original trainable neural network model.

* **Purpose:** A recognition system for binary classification (separating inputs into two classes, A and B).
* **Algorithm:** Implements the Perceptron learning algorithm, which iteratively adjusts weights to find an optimal hyperplane that separates the feature space.
* **Process:** The system is designed to load image patterns, create normalized feature vectors, train on a dataset, and then classify an unknown pattern.

---

### 2. Hopfield Network
**(Branch: `Hopfield`)**

An implementation of a recurrent neural network that functions as a content-addressable (associative) memory.

* **Purpose:** An associative memory system. It can retrieve a complete, stored pattern when presented with a partial or noisy version of it.
* **Algorithm:**
    * Calculates a weight matrix based on a set of "ideal" sample patterns.
    * Binarizes the sample vectors (using values of -1 and 1).
    * An iterative "relaxation" process where the network's state evolves until it stabilizes, ideally converging to one of the stored patterns.
* **Notes:** This model has a limited storage capacity (number of patterns `m` ≈ 0.15 * `n`, where `n` is the number of neurons) and can sometimes converge to incorrect local minima.

---

### 3. Hamming Network
**(Branch: `Heming`)**

An implementation of the Hamming network, a recurrent classification model known for its efficiency and accuracy.

* **Purpose:** An optimal classifier that finds the "closest" matching pattern from its memory, based on the minimum Hamming distance (the number of differing bits).
* **Structure:**
    1.  **Feedforward Layer:** Calculates the initial similarity of the input vector to each stored pattern.
    2.  **Recurrent Layer:** A "winner-take-all" competitive layer. Neurons iteratively suppress each other (via inhibitory connections) until only one neuron, representing the best match, remains active.
* **Advantages over Hopfield:**
    * Requires fewer neurons (one per stored pattern).
    * Generally faster and more accurate.
    * Does not suffer from the same local minima problems as the Hopfield network.

## 📂 Repository Structure

* **`main`**: The main branch, ideally containing the most stable version or an overview.
* **`perceptron`**: Contains the Perceptron implementation.
* **`Hopfield`**: Contains the Hopfield Network implementation.
* **`Heming`**: Contains the Hamming Network implementation.

## 🛠️ Setup & Running

### Prerequisites

* **Core Libraries:** This project relies on [Numpy](https://numpy.org/) for numerical operations and [Streamlit](https://streamlit.io/) for the interactive UI. You can install them via `pip`:
    ```bash
    pip install numpy streamlit
    ```
    (Or by using the `requirements.txt` file, if one is provided in the branch).

* **Data:** The models are designed for image pattern recognition. A good dataset for testing is the [MNIST dataset (JPG format)](https://www.kaggle.com/datasets/scolianni/mnistasjpg), which contains images of handwritten numbers, letters, or symbols.

### Running an Example

1.  Clone the repository:
    ```bash
    git clone https://github.com/BIazeStorm/basic-ai.git
    ```
2.  Change to the project directory:
    ```bash
    cd basic-ai
    ```
3.  Check out the branch you're interested in (e.g., `perceptron`):
    ```bash
    git checkout perceptron
    ```
4.  Run the Streamlit application:
    ```bash
    streamlit run main.py
    ```
    *(Note: The main file might be named `main.py` or `app.py`)*
