# SVD-based Article Recommendation System

This repository contains an article recommendation engine built using interaction data from the IBM Watson Studio platform. It implements several machine learning techniques—including **Singular Value Decomposition (SVD)**—to provide personalized article recommendations based on user behavior.

---

## 🎥 Demo

[View Demo GIF](demo_svd.gif) (13MB)

*Note: The GIF is large and may take a moment to load. For a smoother experience, you can download it and view it locally.*

---

## 📌 Table of Contents

- [🎥 Demo](#-demo)
- [🧩 Introduction](#-introduction)
- [📊 Dataset](#-dataset)
- [🎯 Recommendation Methods](#-recommendation-methods)
- [🚀 Features](#-features)
- [🗂️ Project Structure](#%EF%B8%8F-project-structure)
- [⚙️ Installation](#%EF%B8%8F-installation)
- [⚡ Quick Start](#-quick-start)
- [🌐 Web Application](#-web-application)
- [🙏 Acknowledgements](#-acknowledgements)
- [📄 License](#-license)

---

## 🧩 Introduction

This project explores user-article interactions to generate personalized article recommendations. It supports multiple strategies—from simple popularity-based methods to matrix factorization techniques—and includes an interactive **Streamlit web app** for demonstration and exploration.

---

## 📊 Dataset

The system uses the following datasets:

* `user-item-interactions.csv`: Logs of user interactions with articles.
* `articles_community.csv`: Metadata for each article, including title and description.

---

## 🎯 Recommendation Methods

The project implements the following recommendation strategies:

* **Rank-Based Recommendations**
  Suggests the most popular articles overall.

* **User-User Collaborative Filtering**
  Recommends articles based on similar users' preferences.

* **Content-Based Filtering**
  Uses TF-IDF on article descriptions to recommend similar articles.

* **Matrix Factorization (SVD)**
  Predicts user-article interactions using latent features learned from the interaction matrix.

---

## 🚀 Features

* Personalized article recommendations.
* Multiple algorithmic approaches for comparison.
* Streamlit web interface for easy interaction.
* Notebook analysis with HTML export for review.

---

## 🗂️ Project Structure

```
svd-recommendation-engine/
├── app.py                         # Streamlit web app
├── project_tests.py              # Unit tests
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── data/
│   ├── articles_community.csv
│   └── user-item-interactions.csv
├── top_5.p                       # Top-5 recommendation model
├── top_10.p                      # Top-10 recommendation model
├── top_20.p                      # Top-20 recommendation model
├── user_item_matrix.zip          # Pickled matrix (used for SVD)
├── Recommendations_with_IBM.ipynb # Main Jupyter Notebook
└── Recommendations_with_IBM.html  # Notebook HTML version
```

---

## ⚙️ Installation

### Requirements

* Python 3.8 or later
* `pip` (Python package manager)

---

## ⚡ Quick Start

Clone the repository and run the app:

```bash
git clone https://github.com/omkarbhad/svd-recommendation-engine.git
cd svd-recommendation-engine
pip install -r requirements.txt
unzip user_item_matrix.zip
streamlit run app.py
```

Then visit: [http://localhost:8501](http://localhost:8501)

---

## 🌐 Web Application

### 🔹 Home Page

* Enter a user ID (1–5148) and choose a recommendation model.
* View top-N personalized recommendations.

### 🔹 Browse Articles

* Select any article and get content-based similar suggestions.

### 🔹 Notebook Explorer

* Read the full analysis in the HTML-exported Jupyter notebook.

---

## 🙏 Acknowledgements

* **Udacity** – Data Scientist Nanodegree Program
* **IBM** – For the dataset and project foundation
* **Streamlit** – For the interactive app framework

---

## 📄 License

Licensed under the [MIT License](LICENSE).
