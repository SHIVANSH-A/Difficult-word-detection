# Word Difficulty Prediction Model

## Overview
This project implements a machine learning model to predict the difficulty of words based on various linguistic and statistical features. The model is trained using the `GradientBoostingClassifier` from `sklearn.ensemble` and utilizes datasets containing word frequency, reaction times, and other features.

## Features Used
The following features are used for training:
- **Length**: Number of characters in the word.
- **Log_Freq_HAL**: Logarithm of word frequency in the HAL dataset.
- **I_Mean_RT**: Mean reaction time for word recognition.
- **I_SD**: Standard deviation of reaction times.
- **I_Zscore**: Z-score of reaction times.
- **Obs**: Number of observations recorded for the word.

## Data Preprocessing
1. **Handling Missing Values**: The dataset is cleaned by removing rows with missing values.
2. **Outlier Removal**: Outliers are removed using both Interquartile Range (IQR) and Z-score methods.
3. **Label Assignment**: Words with an `I_Mean_Accuracy` score below `0.64` are labeled as `hard` (1), otherwise as `easy` (0).
4. **Splitting Data**: The dataset is split into training (80%) and testing (20%) sets.

## Model Training
The `GradientBoostingClassifier` is used with the following hyperparameters:
- `n_estimators=100`
- `learning_rate=0.1`
- `max_depth=3`
- `random_state=42`

## Prediction Functions
The model is used to predict the difficulty of:
1. **Single Words**: Given a word, the system either fetches its existing features or estimates them based on a word frequency dataset.
2. **Paragraphs**: The function extracts words from a given text, computes their difficulty, and returns a dictionary mapping words to their predicted difficulty level.

## Example Usage
A paragraph containing multiple words is processed, and each word is classified as either `easy` or `hard` based on the model's prediction.

## Required Libraries
- `pandas`
- `numpy`
- `sklearn`
- `re`

## Datasets
- `WordDifficulty.csv`: Contains word difficulty scores and linguistic features.
- `WordFrequency.csv`: Provides frequency counts for words.

## Running the Code
1. Ensure that all required datasets are available.
2. Install necessary dependencies using `pip install pandas numpy scikit-learn`.
3. Run the script to train the model and predict word difficulties.



