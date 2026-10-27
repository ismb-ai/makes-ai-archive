# anomaly.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

# The following document contains the routines of the sales anomaly detection
# system. We utilize a machine learning approach based on a Generative Adversarial
# Network (GAN) to find anomalies in sales data.

# In sec. 1, we establish procedures for constructing the architecture of the
# Generative Adversarial Network (GAN), including the generator and discriminator
# models. In Section 2, we delineate the primary functions of our algorithm, which
# oversee model training and output collection. Finally, in sec. 3, we present
# the results through printing and plotting within the web interface. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from tensorflow.keras import models, layers, optimizers

# Section 1: Architectural functions

# Build GAN's generator
def build_generator():
    # Create a Sequential model
    model = models.Sequential([
        # Add a dense layer with 16 units and ReLU activation function, expecting input of dimension 2
        layers.Dense(16, activation='relu', input_dim=2),
        # Add another dense layer with 16 units and ReLU activation function
        layers.Dense(16, activation='relu'),
        # Add a dense output layer with 2 units and linear activation function
        layers.Dense(2, activation='linear')
    ])
    # Return the built generator model
    return model

# Buld GAN's discriminator
def build_discriminator():
    # Create a Sequential model
    model = models.Sequential([
        # Add a dense layer with 16 units and ReLU activation function, expecting input of dimension 2
        layers.Dense(16, activation='relu', input_dim=2),
        # Add another dense layer with 16 units and ReLU activation function
        layers.Dense(16, activation='relu'),
        # Add a dense output layer with 1 unit and sigmoid activation function
        layers.Dense(1, activation='sigmoid')
    ])
    # Return the built discriminator model
    return model

# Compile the discriminator model
def compile_discriminator(discriminator):
    # Compile the discriminator model using the Adam optimizer, binary crossentropy loss, and accuracy metric
    discriminator.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Create the combined GAN model
def create_combined_model(generator, discriminator):
    # Set the discriminator's trainable parameter to False
    discriminator.trainable = False
    # Define an input layer for noise with shape (2,)
    z = layers.Input(shape=(2,))
    # Generate data using the generator model
    gen_output = generator(z)
    # Get the discriminator's output by passing the generated data through it
    dis_output = discriminator(gen_output)
    # Create a combined model with input z and output dis_output
    combined = models.Model(z, dis_output)
    # Compile the combined model using the Adam optimizer and binary crossentropy loss
    combined.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    # Return the combined model
    return combined

# Section 2: Engine

# Train the GAN
def train_gan(combined, discriminator, generator, X_train, epochs, batch_size):
    # Iterate through epochs
    for epoch in range(epochs):
        # Randomly sample indices for batch training
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        # Extract real data for training
        real_data = X_train.iloc[idx][['Feature1', 'Feature2']].values
        # Generate random noise
        noise = np.random.normal(0, 1, (batch_size, 2))
        # Generate fake data using the generator
        gen_data = generator.predict(noise)

        # Train the discriminator with real data and ones as labels
        d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        # Train the discriminator with fake data and zeros as labels
        d_loss_fake = discriminator.train_on_batch(gen_data, np.zeros((batch_size, 1)))
        # Calculate the average discriminator loss
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Generate noise for training the combined model
        noise = np.random.normal(0, 1, (batch_size, 2))
        # Create labels indicating that the generated data is real
        valid_y = np.ones((batch_size, 1))
        # Train the combined model
        g_loss = combined.train_on_batch(noise, valid_y)

        # Print loss values at intervals
        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {d_loss[1]}] [G loss: {g_loss}]")

# Detect anomalies using the generator
def detect_anomalies(generator, X_test):
    # Generate noise for the given test data
    noise = np.random.normal(0, 1, (X_test.shape[0], 2))
    # Generate synthetic data using the generator
    gen_data = generator.predict(noise)
    # Extract real data from the test dataset
    real_data = X_test[['Feature1', 'Feature2']].values
    # Calculate anomaly scores by computing the Euclidean distance between real and generated data
    anomalies = np.linalg.norm(real_data - gen_data, axis=1)
    # Add the anomaly scores as a new column to the test dataset
    X_test['AnomalyScore'] = anomalies
    # Return the test dataset with anomaly scores
    return X_test

# Section 3: Output

# Print anomalies
def print_anomalies(anomalies_df):
    # Print a subheader indicating detected anomalies
    st.subheader("Anomalies Detected:")
    # Display relevant columns of the dataframe containing anomalies
    st.write(anomalies_df[['Product_ID', 'Client_ID', 'Sales_Date', 'AnomalyScore']])

# Plot anomalies
def plot_anomalies(X_test):
    # Create a scatter plot with anomaly scores as colors
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_test['Feature1'], X_test['Feature2'], c=X_test['AnomalyScore'], cmap='coolwarm')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Anomaly Detection')
    # Add a colorbar to the plot
    plt.colorbar(scatter, ax=ax, label='Anomaly Score')
    # Display the plot
    st.pyplot(fig)
