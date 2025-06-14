
# 💬 RNN Review Sentiment Analysis

This project implements a **Recurrent Neural Network (RNN)** using TensorFlow/Keras to classify user review sentiments as positive or negative. It uses IMDB movie review data and leverages embedding layers and simple RNN layers for natural language understanding.

---

## 📁 Project Structure

```
📂 RNN_Review_Sentiment/
│
├── 📄 simple_rnn.py           # Text preprocessing and padding, RNN model creation and training
├── 📄 app.py                  # Streamlit app
├── 📄 model.h5                # Trained RNN model
├── 📄 requirements.txt        # Dependencies
└── 📄 README.md               # Project documentation
```

---

## 📊 Dataset Overview

- **Dataset**: IMDB Movie Reviews
- **Size**: 25,000 training and 25,000 testing samples
- **Classes**:  
  - `1` → Positive Sentiment  
  - `0` → Negative Sentiment

---

## 🛠️ Preprocessing

- Tokenized review text  
- Applied padding/truncating to ensure fixed input length  
- Converted text to sequences using `Tokenizer`  
- Saved tokenizer word index with `pickle`

---

## 🧠 Model Architecture

- Embedding Layer → Simple RNN Layer → Dense Output Layer  
- Activation: ReLU + Sigmoid  
- Loss: Binary Crossentropy  
- Optimizer: Adam  
- Metric: Accuracy

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📈 Evaluation

- Accuracy and loss are printed per epoch  
- Model performance is evaluated on the test set  
- Final model saved as `model.h5`

---

