import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the 'mpg' dataset from seaborn
df = sns.load_dataset('mpg').dropna()  # Drop missing values

# Select relevant columns (we'll use 'horsepower' to predict 'mpg')
df = df[['horsepower', 'mpg']]

# Drop rows with missing values
df = df.dropna()

# Split the data into independent (X) and dependent (y) variables
X = df[['horsepower']]
y = df['mpg']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a LinearRegression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Plot the results
plt.scatter(X_test, y_test, color='blue', label='Actual mpg')
plt.scatter(X_test, y_pred, color='red', label='Predicted mpg')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.title('Linear Regression: Predicting MPG based on Horsepower')
plt.legend()
plt.show()