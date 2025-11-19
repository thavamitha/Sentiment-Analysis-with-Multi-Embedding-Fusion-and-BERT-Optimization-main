# Sentiment Analysis with Multi-Embedding Fusion and BERT Optimization

This project implements a progressive pipeline for sentiment analysis, starting from traditional models like TF-IDF and advancing to powerful contextual embedding techniques like BERT. By evaluating each method on a custom-labeled dataset of Google Play Store reviews, this project highlights the strengths, limitations, and improvements across various NLP techniques.

## 📌 Project Highlights

- Comparative analysis of **TF-IDF**, **Word2Vec**, **GloVe**, **FastText**, and **BERT** models.
- Final pipeline integrates **BERT embeddings** with **Support Vector Machine (SVM)** for high accuracy.
- Custom preprocessing pipeline that preserves negation and handles noisy, real-world text data.
- Misclassification analysis for every model to understand model limitations and error patterns.
- Achieved **97.5% accuracy** using BERT + SVM on real-world review sentiment classification.

## 📂 Project Structure

```
.
├── cleaned_dataset.csv             # Preprocessed dataset used for training
├── modified_dataset.csv            # Original labeled dataset
├── tfidf_misclassified_data.csv    # Misclassifications (TF-IDF)
├── word2vec_misclassified_data.csv
├── glove_misclassified_data.csv
├── fasttext_misclassified_data.csv
├── bert_misclassified_data.csv
├── models/
│   ├── tfidf_lr_model.pkl
│   ├── word2vec_lr_model.pkl
│   ├── glove_lr_model.pkl
│   ├── fasttext_lr_model.pkl
│   └── bert_svm_model.pkl
├── src/
│   ├── preprocessing.py
│   ├── training_pipeline.py
│   └── evaluation.py
├── README.md
└── requirements.txt
```

## 🧠 Models Compared

| Model                  | Accuracy | F1-Score | Strengths                                     | Limitations                            |
|------------------------|----------|----------|-----------------------------------------------|-----------------------------------------|
| TF-IDF + Logistic Reg. | 83%      | 0.83     | Fast, interpretable, strong baseline          | Ignores context, fails on negation      |
| Word2Vec + LR          | 52%      | 0.40     | Captures semantic similarity                  | No context, fails on short texts        |
| GloVe + LR             | 81%      | 0.81     | Uses global co-occurrence statistics          | Still context-agnostic                  |
| FastText + LR          | 85%      | 0.85     | Handles rare/misspelled words(subword n-grams)| Cannot model sentence context           |
| BERT + SVM             | 97.5%    | 0.97     | Deep contextual understanding, robust to noise| Requires GPU, high computational cost   |

## 🛠️ Preprocessing Pipeline

- Lowercasing
- URL & HTML tag removal
- Punctuation stripping
- Contraction expansion
- Stopword removal (negations preserved)
- Tokenization
- Lemmatization
- Duplicate & null record removal

## 🧪 Evaluation Strategy

- **Accuracy**, **Precision**, **Recall**, **F1-score** calculated per model
- Deep misclassification analysis per technique
- Focused on how models handle negation, sarcasm, OOV words, and contextual ambiguity

## 📊 Model Architecture

![image](https://github.com/user-attachments/assets/49a8c42a-e3ea-4f85-baed-d888de5ff7cf)

## 📈 Final Results

- The **BERT + SVM** hybrid model demonstrated **state-of-the-art performance** by using [CLS] token embeddings as input to an SVM classifier.
- Common issues like sarcasm, idioms, and negations were better handled with contextual embeddings.

## 🔍 Future Work

- Fine-tuning BERT on domain-specific data (e.g., BERTweet for social reviews)
- Implementing ensemble models combining traditional + deep learning
- Adding real-time inference and deployment (Flask/Streamlit UI)
- Augmenting data with paraphrased and sarcastic samples
 

## 📦 Requirements

```bash
transformers==4.x
scikit-learn==1.x
nltk==3.x
pandas==1.x
numpy==1.x
gensim==4.x
fasttext
contractions
```

## 📜 License

This project is part of the B.Tech curriculum at Shiv Nadar University and is intended for academic and educational purposes.



