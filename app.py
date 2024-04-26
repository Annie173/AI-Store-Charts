from flask import Flask, render_template
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import base64
from io import BytesIO

app = Flask(__name__)

@app.route('/')
def index():
    # Load the dataset
    data = pd.read_csv('data/Stores.csv')

    # Prepare features (X) and target variable (y)
    X = data[['Store_Area', 'Items_Available', 'Daily_Customer_Count']]
    y = data['Store_Sales']

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Build the neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)  # Output layer for regression task
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print("Mean Squared Error on Test Data:", loss)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print("Root Mean Squared Error on Test Data:", rmse)

    # Display basic statistics of the dataset
    print(data.describe())

    # Pairplot to visualize relationships between variables
    pairplot_img = BytesIO()
    sns.pairplot(data)
    plt.savefig(pairplot_img, format='png')
    pairplot_img.seek(0)
    pairplot_encoded = base64.b64encode(pairplot_img.getvalue()).decode('utf-8')
    plt.clf()

    # Correlation heatmap
    heatmap_img = BytesIO()
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.savefig(heatmap_img, format='png')
    heatmap_img.seek(0)
    heatmap_encoded = base64.b64encode(heatmap_img.getvalue()).decode('utf-8')
    plt.clf()

    # Distribution of Store Sales
    sales_distribution_img = BytesIO()
    plt.figure(figsize=(8, 6))
    sns.histplot(data['Store_Sales'], kde=True)
    plt.savefig(sales_distribution_img, format='png')
    sales_distribution_img.seek(0)
    sales_distribution_encoded = base64.b64encode(sales_distribution_img.getvalue()).decode('utf-8')
    plt.clf()

    return render_template('index.html', pairplot_encoded=pairplot_encoded, heatmap_encoded=heatmap_encoded, 
                           sales_distribution_encoded=sales_distribution_encoded)

if __name__ == '__main__':
    app.run(port=8000)