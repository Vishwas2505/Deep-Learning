import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

st.title("🚗 Car Price Prediction using Deep Learning")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("CarPrice_dataset.csv")
    return df

df = load_data()

st.subheader("Dataset Preview")
st.write(df.head())

# -----------------------------
# Data Cleaning
# -----------------------------
df = df.drop_duplicates()

num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    df[col].fillna(df[col].mean(), inplace=True)

cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# convert types
if 'horsepower' in df.columns:
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

# -----------------------------
# Prepare Data
# -----------------------------
X = df.drop("price", axis=1)
y = df["price"]

categorical_cols = X.select_dtypes(include=['object','category']).columns

X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Build Model
# -----------------------------
model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# -----------------------------
# Train Model
# -----------------------------
with st.spinner("Training Model..."):
    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=0
    )

st.success("Model Training Completed")

# -----------------------------
# Model Evaluation
# -----------------------------
y_pred = model.predict(X_test).flatten()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Evaluation")

st.write("MAE:", mae)
st.write("MSE:", mse)
st.write("RMSE:", rmse)
st.write("R2 Score:", r2)

# -----------------------------
# Prediction Section
# -----------------------------
st.subheader("Predict Car Price")

brand = st.text_input("Brand", "Toyota")
fueltype = st.selectbox("Fuel Type", ["petrol","diesel"])
enginesize = st.number_input("Engine Size", 800, 5000, 1800)
mileage = st.number_input("Mileage (km/l)", 5, 40, 15)
transmission = st.selectbox("Transmission", ["manual","automatic"])

if st.button("Predict Price"):

    new_car = pd.DataFrame({
        "brand":[brand],
        "fueltype":[fueltype],
        "enginesize":[enginesize],
        "mileage":[mileage],
        "transmission":[transmission]
    })

    new_car_encoded = pd.get_dummies(new_car)

    new_car_encoded = new_car_encoded.reindex(columns=X.columns, fill_value=0)

    new_car_scaled = scaler.transform(new_car_encoded)

    predicted_price = model.predict(new_car_scaled)[0][0]

    st.success(f"Predicted Car Price = ₹{predicted_price:,.0f}")