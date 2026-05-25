import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load Dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Select Features
X = df[["Pclass", "Age", "Fare"]]

# Fill Missing Values
X["Age"] = X["Age"].fillna(X["Age"].mean())

# Target
y = df["Survived"]

# Normalize
X = X / X.max()

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Build Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(
        8,
        activation='relu',
        input_shape=(3,)
    ),
    tf.keras.layers.Dense(
        4,
        activation='relu'
    ),
    tf.keras.layers.Dense(
        1,
        activation='sigmoid'
    )
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train Model
model.fit(
    X_train,
    y_train,
    epochs=50
)

# Save Model
model.save("titanic_ann_model.h5")

print("Model saved successfully!")