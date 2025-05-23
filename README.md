# 🧠 Heart Disease Prediction — Neural Network from Scratch

This project implements a fully connected **neural network from scratch** (without using any ML libraries like Keras, TensorFlow, or PyTorch) to predict the presence of heart disease using the [Cleveland Heart Disease dataset](https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland).

> 💡 GitHub Repo: [M21-22/BCNN](https://github.com/M21-22/BCNN)

---

## 🔧 Stack

- Python
- NumPy
- pandas
- scikit-learn (for preprocessing, metrics)

---

## 🧠 Network Architecture

- **Input Layer**: 13 features
- **Hidden Layer 1**: 10 neurons + sigmoid
- **Hidden Layer 2**: 7 neurons + sigmoid
- **Hidden Layer 3**: 4 neurons + sigmoid
- **Output Layer**: 1 neuron + sigmoid (for binary classification)

- **Loss Function**: Binary Cross-Entropy  
- **Optimizer**: Manual Gradient Descent (with backpropagation)

---

## 📊 Performance

Model evaluation on the test set:

```
             precision    recall  f1-score   support

           0       0.94      0.88      0.91        33
           1       0.87      0.93      0.90        28

    accuracy                           0.90        61
   macro avg       0.90      0.90      0.90        61
weighted avg       0.90      0.90      0.90        61
```

---

## 🚀 How to Run

1. **Clone the repository**

```bash
git clone https://github.com/M21-22/BCNN.git
cd BCNN
```

2. **Install required packages**

```bash
pip install numpy pandas scikit-learn
```

3. **Download the dataset**

Download the [Cleveland Heart Disease dataset](https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland) and place the `heart.csv` file in the project root directory.

4. **Train the model**

```bash
python train.py
```

---

## 🤖 Why Build from Scratch?

This project was built for educational purposes to deeply understand:
- Matrix operations behind neural networks
- How forward and backward propagation work
- How gradients are computed and used to update weights manually

---

## 📚 References

- [Kaggle Dataset](https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland)

---

## 🪪 License

No License — use, modify, and share freely.

---

## 🙌 Author

Developed by [@M21-22](https://github.com/M21-22)
