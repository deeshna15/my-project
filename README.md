# my-project
Fake News Detection project that classifies news articles as real or fake using deep learning models in Python.
# 📰 Fake News Detection using LSTM

This project uses Natural Language Processing (NLP) and a deep learning LSTM model to classify news headlines as **Real** or **Fake**.

---

## 📌 Table of Contents

- [Demo](#-demo)
- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [Setup Instructions](#-setup-instructions)
- [How It Works](#-how-it-works)
- [Model Architecture](#-model-architecture)
- [Future Improvements](#-future-improvements)

---

## 🎥 Demo
Fake News Prediction
![image](https://github.com/user-attachments/assets/8c84c81a-3a11-45f9-83b9-aa777052cc9e)
Real News Prediction

![image](https://github.com/user-attachments/assets/22c986bf-8325-422e-a08e-cb96e536ccaa)



---

## 🚀 Features

- Binary classification of news titles as Real or Fake
- Pretrained GloVe embeddings
- LSTM + CNN-based hybrid model
- Web interface using FastAPI

---

## 🧠 Technologies Used

| Category       | Stack                     |
|----------------|---------------------------|
| Language       | Python                    |
| ML Framework   | TensorFlow, Keras         |
| Embeddings     | GloVe 50d                 |
| Web Framework  | FastAPI                   |
| Frontend       | HTML, CSS                 |
| IDE            | Google Colab, VS Code     |

---

## ⚙️ Setup Instructions
1.Clone the Repository

2.Install Dependencies

pip install -r requirements.txt

3.Download GloVe Embeddings

Download glove.6B.50d.txt from https://nlp.stanford.edu/projects/glove/

Place the file in the project root directory.

4.Train or Load the Model

Run the Jupyter notebook to train the model.

Save the model and tokenizer as model.h5 and tokenizer.pkl.

5.Start the FastAPI Server
uvicorn app:app --reload

## 🔍 How It Works

- User enters a news headline  
- Input is tokenized and padded  
- The model predicts the probability of the news being real  
- Prediction is shown on the web interface as **Real** or **Fake**

---

## 🧬 Model Architecture

- Embedding Layer with pretrained GloVe vectors (50d)  
- Dropout Layer to prevent overfitting  
- 1D Convolutional Layer to extract features  
- MaxPooling Layer  
- LSTM Layer to capture sequence dependencies  
- Dense Output Layer with sigmoid activation  

---

## 🔮 Future Improvements

- Improve prediction accuracy with better embeddings (e.g., GloVe 100d or BERT)  
- Deploy using Docker or cloud services  
- Support full article input instead of just titles  
- Add user feedback and learning loop  



