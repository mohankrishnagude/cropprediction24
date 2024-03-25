import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

data = pd.read_csv("Temphum.csv")

X = data[['Temperature', 'Humidity', 'Moisture']]
y = data['WaterRequirement']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model as a pickle file
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)
