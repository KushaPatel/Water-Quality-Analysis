# Water Quality Analysis
This project involves the analysis of water quality data to predict whether water is potable (safe to drink) or not. The dataset contains various water quality parameters, and the goal is to classify the water's potability based on these features using machine learning models.

## Project Overview
The dataset used in this project contains features such as pH, hardness, turbidity, sulfate concentration, and others. The target variable is `Potability`, which indicates whether the water is potable (1) or not (0).

This project performs the following steps:
1. Data cleaning (handling missing values and removing outliers).
2. Data preprocessing (scaling features and encoding).
3. Model training using Logistic Regression and Random Forest classifiers.
4. Hyperparameter tuning for Random Forest using GridSearchCV.
5. Model evaluation with metrics like confusion matrix, classification report, ROC curve, and cross-validation accuracy.
6. Visualizations including boxplots, histograms, correlation heatmaps, and feature importance.

## Files in the Project
- `Dataset/water_potability.csv`: The dataset file.
- `Code/water_qualitys.py`: The main Python script for loading, processing the data, and training models.

## Setup Instructions
### Prerequisites
1. **Python 3.x** must be installed.
2. Install required dependencies.

##Running the Code
1.Clone the Repository:
git clone https://github.com/KushaPatel/Water-Quality-Analysis.git

2.Navigate to the Project Directory:
cd Water-Quality-Analysis

3.Activate the Virtual Environment (if applicable):
For Windows:
.\env\Scripts\activate
For macOS/Linux:
source env/bin/activate

4.Run the Python Script: After installing dependencies and setting up the environment, run the Python script water_qualitys.py to perform the analysis:
python Code/water_qualitys.py



