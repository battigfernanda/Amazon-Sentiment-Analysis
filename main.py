import bz2
import pandas as pd
import numpy as np
import string
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import shap
import multiprocessing
import os
import gc

# Convert GloVe to Word2Vec format
def convert_glove_to_word2vec(glove_input_file, word2vec_output_file):
    glove2word2vec(glove_input_file, word2vec_output_file)

# Optimized Decompression and Loading
def decompress_bz2_to_df(file_path, sample_fraction=0.01):
    chunks = []
    chunk_size = 5000
    with bz2.open(file_path, 'rt') as bz_file:
        while True:
            try:
                chunk = pd.read_csv(bz_file, sep='\t', header=None, nrows=chunk_size, on_bad_lines='skip')
                if chunk.empty:
                    break
                if sample_fraction < 1.0:
                    chunk = chunk.sample(frac=sample_fraction)
                chunks.append(chunk)
                print(f"Processed a chunk of size {chunk_size}")
            except pd.errors.EmptyDataError:
                break
    data = pd.concat(chunks, ignore_index=True)
    return data

# Text Cleaning
def clean_text(text):
    text = text.replace('__label__2 ', '').replace('__label__1 ', '')
    return text.lower().translate(str.maketrans('', '', string.punctuation))

# Data Preprocessing
def preprocess_data(data):
    data.columns = ['raw_text']
    data['text'] = data['raw_text'].apply(clean_text)
    data['label'] = data['raw_text'].apply(lambda x: 1 if '__label__2' in x else 0)
    data.drop(columns=['raw_text'], inplace=True)
    return data

# Load GloVe Efficiently (50d)
def load_glove_model(glove_file_path):
    print("Converting GloVe to Word2Vec format...")
    word2vec_glove_file = glove_file_path + '.word2vec'
    if not os.path.exists(word2vec_glove_file):
        convert_glove_to_word2vec(glove_file_path, word2vec_glove_file)
    print("Loading GloVe model in Word2Vec format...")
    glove_model = KeyedVectors.load_word2vec_format(word2vec_glove_file, binary=False)
    print("GloVe model loaded.")
    return glove_model

# Convert text to GloVe vector (50d)
def text_to_vector(text, glove_model, vector_size=50):
    words = text.split()
    vectors = np.array([glove_model[word] for word in words if word in glove_model])
    return np.mean(vectors, axis=0) if len(vectors) > 0 else np.zeros(vector_size)

# Check class imbalance and apply SMOTE if needed
def balance_classes(X, y):
    if max(Counter(y).values()) / min(Counter(y).values()) > 1.5:
        print("Applying SMOTE to balance classes...")
        smote = SMOTE(sampling_strategy=0.8)  # Adjusted strategy for less noise
        X, y = smote.fit_resample(X, y)
    return X, y

# Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {title}')
    plt.show()

# Plot ROC Curve
def plot_roc_curve(model, X_test, y_test, title):
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title(f'ROC Curve for {title}')
    plt.legend(loc="lower right")
    plt.show()

# Main Code
if __name__ == "__main__":
    train_path = '/Users/fernandabattig/Desktop/Amazon Sentiment Anaysis/archive-2/train.ft.txt.bz2'
    test_path = '/Users/fernandabattig/Desktop/Amazon Sentiment Anaysis/archive-2/test.ft.txt.bz2'
    glove_path = '/Users/fernandabattig/Desktop/Amazon Sentiment Anaysis/glove/glove.6B.50d.txt'  # Use 50d GloVe

    train_data = decompress_bz2_to_df(train_path, sample_fraction=0.01)
    test_data = decompress_bz2_to_df(test_path, sample_fraction=0.01)

    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    glove_model = load_glove_model(glove_path)

    # Parallelized Text Vectorization
    with multiprocessing.Pool() as pool:
        train_data['vector'] = pool.starmap(text_to_vector, [(text, glove_model) for text in train_data['text']])
        test_data['vector'] = pool.starmap(text_to_vector, [(text, glove_model) for text in test_data['text']])

    X_train = np.vstack(train_data['vector'].values).astype('float32')
    y_train = train_data['label'].values
    X_test = np.vstack(test_data['vector'].values).astype('float32')
    y_test = test_data['label'].values

    # Random Forest with RandomizedSearchCV
    rf = RandomForestClassifier()
    param_dist = {'n_estimators': [100, 200], 'max_depth': [10, 20]}  # Simplified params for speed
    grid_rf = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=3, cv=3, n_jobs=-1)
    grid_rf.fit(X_train, y_train)

    predictions_rf = grid_rf.predict(X_test)
    print("Optimized RF Accuracy:", accuracy_score(y_test, predictions_rf))
    plot_confusion_matrix(y_test, predictions_rf, "Random Forest")
    plot_roc_curve(grid_rf, X_test, y_test, "Random Forest")

    # SVM Model Training
    print("Training SVM...")
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train, y_train)
    predictions_svm = svm.predict(X_test)
    print("SVM Accuracy:", accuracy_score(y_test, predictions_svm))
    plot_confusion_matrix(y_test, predictions_svm, "SVM")
    plot_roc_curve(svm, X_test, y_test, "SVM")

    # Efficient SHAP Calculation
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(grid_rf.best_estimator_)
    X_sample = shap.sample(X_train, 200)  # Reduced to 200 for speed
    shap_values = explainer.shap_values(X_sample)ß
    shap.summary_plot(shap_values, X_sample)

    # Efficient Memory Management
    del glove_model, train_data, test_data
    gc.collect()
