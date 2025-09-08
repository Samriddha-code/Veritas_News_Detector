# Fake News Prediction Project

**Live Demo:** [**Click here to try the app!**](https://veritasnewsdetector-nnxb7kapp6jspfutb7xd28j.streamlit.app/)

This project contains a machine learning model built to classify news articles as either "Real" or "Fake". It utilizes Natural Language Processing (NLP) techniques to process text data and a Logistic Regression model to perform the classification, achieving an accuracy of over 98%.

---

## 1. Project Goal

The primary objective of this project is to develop an accurate and efficient machine learning model to classify news articles as either **Real News** or **Fake News**. This addresses the growing problem of misinformation by providing an automated solution for content verification.

---

## 2. Datasets Used

The model was trained on a combined and balanced dataset from three sources to ensure a diverse and robust training corpus:

* **`Fake.csv`**: A collection of articles known to be fake news.
* **`True.csv`**: A collection of legitimate news articles from reliable sources.
* **`BBC News Train.csv`**: An additional set of verified news articles from the BBC to enrich the "real news" data.

---

## 3. Project Methodology

The project followed a standard machine learning workflow:

* **Data Preparation**: The datasets were merged, and a single `content` field was created from the title and text. To prevent model bias, the dataset was balanced by down-sampling the majority class, ensuring an equal number of real and fake articles.

* **Text Preprocessing (NLP)**: Before training, the text was cleaned and normalized using Natural Language Processing techniques:
  1. **Lowercasing**: All text was converted to lowercase.
  2. **Cleaning**: Punctuation and numbers were removed.
  3. **Stopword Removal**: Common words with little semantic value (e.g., "the", "a", "is") were filtered out.
  4. **Stemming**: Words were reduced to their root form (e.g., "running" -> "run") to group related words.

* **Feature Extraction**: The cleaned text was converted into numerical data using the `TfidfVectorizer`, which calculates a score for each word based on its frequency in an article and its rarity across all articles.

* **Model Training**: A `Logistic Regression` classifier was trained on the processed numerical data. This model was chosen for its excellent performance and efficiency in text classification tasks.

---

## 4. Results

The final model demonstrated strong performance on the unseen test data.

* **Accuracy Score**: `98.37%`

This high accuracy indicates that the model is very effective at distinguishing between real and fake news.

---
