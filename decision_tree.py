import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the 'mpg' dataset from seaborn
df = sns.load_dataset('mpg').dropna()  # Drop missing values

# Select relevant columns (we'll use 'horsepower' to predict 'mpg' class)
df = df[['horsepower', 'mpg']]

# Create a binary classification target: high mpg (>= median) vs low mpg (< median)
median_mpg = df['mpg'].median()
df['mpg_class'] = (df['mpg'] >= median_mpg).astype(int)

# Split the data into independent (X) and dependent (y) variables
X = df[['horsepower']]
y = df['mpg_class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a DecisionTreeClassifier model
model = DecisionTreeClassifier(random_state=42)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=['horsepower'], class_names=['Low mpg', 'High mpg'], filled=True)
plt.title('Decision Tree: Predicting MPG Class based on Horsepower')
plt.show()
