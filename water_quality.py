import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay

# Load Dataset
df = pd.read_csv('../Dataset/water_potability.csv')
print(df.head())
print(df.shape)

# Checking for missing values
print(df.isnull().sum())

# Filling missing values
df.fillna(df.mean(), inplace=True)
print(df.describe())

# Data Type Conversion
df = df.astype({
    'ph': 'float64',
    'Hardness': 'float64',
    'Solids': 'float64',
    'Chloramines': 'float64',
    'Sulfate': 'float64',
    'Conductivity': 'float64',
    'Organic_carbon': 'float64',
    'Trihalomethanes': 'float64',
    'Turbidity': 'float64',
    'Potability': 'int64'
})

# Remove Outliers
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

for column in df.columns[:-1]:
    df = remove_outliers(df, column)

# Feature Scaling
scaler = StandardScaler()
df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])

# Splitting Data
X = df.drop('Potability', axis=1)
y = df['Potability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Cross Validation
log_scores = cross_val_score(log_model, X, y, cv=5, scoring='accuracy')
rf_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')

print(f'Logistic Regression Cross-Validation Accuracy: {log_scores.mean():.2f} ± {log_scores.std():.2f}')
print(f'Random Forest Cross-Validation Accuracy: {rf_scores.mean():.2f} ± {rf_scores.std():.2f}')

# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

# Smaller parameter grid to make it faster
param_grid = {
    'n_estimators': [50, 100],  # instead of [50, 100, 200]
    'max_depth': [None, 10],    # instead of [None, 10, 20, 30]
    'min_samples_split': [2, 5],  # instead of [2, 5, 10]
    'min_samples_leaf': [1, 2]    # instead of [1, 2, 4]
}

# Use all CPU cores with n_jobs = -1
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Fit the model
grid_search.fit(X_train, y_train)

print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Cross-Validation Score: {grid_search.best_score_:.2f}')

best_rf_model = grid_search.best_estimator_
best_rf_model.fit(X_train, y_train)
y_pred_best_rf = best_rf_model.predict(X_test)

print("Tuned Random Forest Classification Report:")
print(classification_report(y_test, y_pred_best_rf))

# Boxplots
plt.figure(figsize=(11, 6))
for i, column in enumerate(df.columns[:-1], 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x='Potability', y=column, data=df)
    plt.title(f'Boxplot of {column}')
plt.tight_layout()
plt.show()

# Feature Importance
importances = best_rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(12, 6))
plt.title('Feature Importance')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# Histograms
plt.figure(figsize=(10, 6))
for i, column in enumerate(df.columns[:-1], 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Pairplot
sns.pairplot(df, hue='Potability')
plt.show()

# Confusion Matrices
cm_log = confusion_matrix(y_test, y_pred_log)
ConfusionMatrixDisplay(confusion_matrix=cm_log).plot()
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

cm_rf = confusion_matrix(y_test, y_pred_rf)
ConfusionMatrixDisplay(confusion_matrix=cm_rf).plot()
plt.title('Confusion Matrix - Random Forest')
plt.show()

# ROC Curves
y_pred_prob_log = log_model.predict_proba(X_test)[:, 1]
fpr_log, tpr_log, _ = roc_curve(y_test, y_pred_prob_log)
roc_auc_log = roc_auc_score(y_test, y_pred_prob_log)

y_pred_prob_rf = rf_model.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_prob_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {roc_auc_log:.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
