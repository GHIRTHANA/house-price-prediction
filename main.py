import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("House Price Prediction Dataset.csv")

# Convert Yes/No columns into numbers
data["Garage"] = data["Garage"].map({
    "Yes": 1,
    "No": 0
})

# Convert text columns to numeric
data["Location"] = data["Location"].astype("category").cat.codes
data["Condition"] = data["Condition"].astype("category").cat.codes

# Input (X)
x = data[[
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "YearBuilt",
    "Location",
    "Condition",
    "Garage"
]]

# Output (Y)
y = data["Price"]

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(x_train, y_train)

# Predict test data
y_pred = model.predict(x_test)

# Accuracy
score = r2_score(y_test, y_pred)
print("R2 Score:", score)

# Predict new house price
new_house = [[1500, 3, 2, 2, 2018, 1, 1, 1]]

prediction = model.predict(new_house)

print("Predicted Price:", prediction[0])