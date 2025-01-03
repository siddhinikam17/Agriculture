import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load the dataset
data = pd.read_csv('data.csv')

# Split the dataset into features (x) and target variable (y)
x = data.drop('label', axis=1)
y = data['label']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# Build and train the logistic regression model
model = LogisticRegression()
model.fit(x_train, y_train)

# Evaluate the model's performance
from sklearn.metrics import classification_report

y_pred = model.predict(x_test)
cr = classification_report(y_test, y_pred)
print(cr)

# Save the model using pickle
pickle.dump(model, open('model1.pkl', 'wb'))
