# Generative Adversarial Network for Stock-out Detection
#
# Igor Mol <igor.mol@makes.ai>
#
# In time series anomaly detection, Generative Adversarial Networks (GANs) can
# be employed to model the normal behavior of a system. During training, the GAN
# learns the underlying patterns in the time series data, and anomalies are
# identified by observing deviations from the learned distribution. The generator
# acts as an anomaly detector by generating synthetic data, and instances where
# real data significantly differs from the generated samples are flagged as
# anomalies.

# For stock-out detection in sales inventory data, GANs can be applied to
# capture the regular sales patterns and identify unusual occurrences indicative
# of potential stock-outs. The generator models the normal distribution of
# sales, and deviations from this learned pattern highlight instances where
# product availability may be compromised.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def load_data(file_path):
    return pd.read_csv(file_path)

# This function preprocesses the input DataFrame (df) by handling missing values
# and scaling the specified features using Min-Max scaling.

def preprocess_data(df, features):
    # Drop rows with missing values in the specified features
    df = df.dropna(subset=features)

    # Initialize Min-Max scaler
    scaler = MinMaxScaler()

    # Apply Min-Max scaling to the selected features
    df[features] = scaler.fit_transform(df[features])

    # Return the preprocessed DataFrame
    return df

# This function defines hyperparameters for the GAN model based on the provided
# features. It sets values for the latent dimension, output dimension, input
# dimension, number of training epochs, and batch size.

def define_hyperparameters(features):
    # Latent dimension for random noise input to the generator
    latent_dim = 10

    # Output dimension based on the number of provided features
    output_dim = len(features)

    # Input dimension based on the number of provided features
    input_dim = len(features)

    # Number of training epochs
    epochs = 10

    # Batch size for training
    batch_size = 64

    # Return the defined hyperparameters
    return latent_dim, output_dim, input_dim, epochs, batch_size

# This function builds the generator model for the GAN. It takes the latent
# dimension and output dimension as input parameters and returns the generator
# model.

def build_generator(latent_dim, output_dim):
    # Initialize a Sequential model for the generator
    model = Sequential()

    # Add a dense layer with ReLU activation for the latent input
    model.add(Dense(16, input_dim=latent_dim, activation='relu'))

    # Add batch normalization to improve training stability
    model.add(BatchNormalization())

    # Add another dense layer with ReLU activation
    model.add(Dense(32, activation='relu'))

    # Add batch normalization for stability
    model.add(BatchNormalization())

    # Add the output layer with sigmoid activation
    model.add(Dense(output_dim, activation='sigmoid'))

    # Return the built generator model
    return model

# This function builds the discriminator model for the GAN. It takes the input
# dimension as a parameter and returns the compiled discriminator model.

def build_discriminator(input_dim):
    # Initialize a Sequential model for the discriminator
    model = Sequential()

    # Add a dense layer with LeakyReLU activation for the input
    model.add(Dense(32, input_dim=input_dim, activation=LeakyReLU(alpha=0.2)))

    # Add batch normalization for stability
    model.add(BatchNormalization())

    # Add another dense layer with LeakyReLU activation
    model.add(Dense(16, activation=LeakyReLU(alpha=0.2)))

    # Add batch normalization for stability
    model.add(BatchNormalization())

    # Add the output layer with sigmoid activation for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model with Adam optimizer and binary crossentropy loss
    model.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')

    # Return the built and compiled discriminator model
    return model

# This function builds and compiles the GAN (Generative Adversarial Network)
# model by combining the generator and discriminator models. The discriminator
# is set to be non-trainable during the GAN training process.

def build_gan(generator, discriminator, latent_dim):
    # Set the discriminator to be non-trainable in the GAN model
    discriminator.trainable = False

    # Define input for the GAN model based on the latent dimension
    gan_input = Input(shape=(latent_dim,))

    # Generate synthetic data using the generator model
    x = generator(gan_input)

    # Use the discriminator to classify the generated data
    gan_output = discriminator(x)

    # Create the GAN model by connecting the generator and discriminator
    gan = Model(gan_input, gan_output)

    # Compile the GAN model with Adam optimizer and binary crossentropy loss
    gan.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')

    # Return the built and compiled GAN model
    return gan

# This function trains a Generative Adversarial Network (GAN) with the provided
# generator, discriminator, and GAN models. It uses a given DataFrame (df)
# containing the training data, specific features, and other hyperparameters.
# The training process involves epochs, batch size, and latent dimension for
# generating random noise.

def train_gan(generator, discriminator, gan, df, features, latent_dim, epochs,
              batch_size):
    # Iterate through training epochs
    for epoch in range(epochs):
        # Generate random noise for the generator
        noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
        # Generate synthetic data using the generator
        generated_data = generator.predict(noise)

        # Sample real data from the DataFrame
        real_data = df[features].values[np.random.randint(0, df.shape[0],
                                                           batch_size)]

        # Create labels for real and fake data for the discriminator
        labels_real = np.ones((batch_size, 1))
        labels_fake = np.zeros((batch_size, 1))

        # Train the discriminator on real and fake data separately
        d_loss_real = discriminator.train_on_batch(real_data, labels_real)
        d_loss_fake = discriminator.train_on_batch(generated_data, labels_fake)

        # Calculate the overall discriminator loss
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Generate new random noise for the generator
        noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
        # Create labels for the GAN (discriminator should predict these as real)
        labels_gan = np.ones((batch_size, 1))

        # Train the GAN (generator) using the combined model
        g_loss = gan.train_on_batch(noise, labels_gan)

        # Print training progress every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} | Discriminator Loss: {d_loss} | Generator Loss: {g_loss}")

# This function performs anomaly detection on the input DataFrame (df) using a
# pre-trained generator model. Anomalies, or stock-outs, are detected based on
# deviations between the real data and data generated by the generator. It
# returns a DataFrame containing the detected stock-outs.

def anomaly_detection(df, features, generator, latent_dim):
    # Generate synthetic data using the pre-trained generator
    num_synthetic_samples = df.shape[0]
    noise = np.random.normal(0, 1, size=(num_synthetic_samples, latent_dim))
    generated_data = generator.predict(noise)

    # Calculate Mean Squared Error (MSE) between real and generated data
    mse = np.mean(np.power(df[features].values - generated_data, 2), axis=1)

    # Set a threshold for anomaly detection using the 95th percentile of MSE
    threshold = np.percentile(mse, 95)

    # Identify stock-outs by comparing MSE to the threshold
    stock_outs = df[mse > threshold]

    # Return the DataFrame containing detected stock-outs
    return stock_outs

# This function performs k-means clustering on the input DataFrame (df) using
# the specified features. The number of clusters can be optionally set, with a
# default value of 3. It returns the original DataFrame with an additional
# 'Cluster' column indicating the assigned cluster for each data point.

def kmeans_clustering(df, features, num_clusters=3):
    # Initialize the k-means model with the given number of clusters
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)

    # Perform k-means clustering and assign cluster labels to the DataFrame
    df['Cluster'] = kmeans.fit_predict(df[features])

    # Return the DataFrame with cluster labels
    return df

# This function labels the clusters identified in the input DataFrame (df)
# based on the 'Cluster' column. It assigns cluster labels such as 'Low',
# 'Average', and 'High-profitability' using specified bins. The result is an
# updated DataFrame with an additional 'Cluster_Label' column.

def label_clusters(df):
    # Define bins and labels for cluster categorization
    bins = [-np.inf, 1, 2, np.inf]
    labels = ['Low', 'Average', 'High-profitability']

    # Use pd.cut to categorize clusters based on the 'Cluster' column
    df['Cluster_Label'] = pd.cut(df['Cluster'], bins=bins, labels=labels)

    # Return the DataFrame with cluster labels
    return df

# This function generates a scatter plot to visualize the clusters in the input
# DataFrame (df) based on the specified features. Each data point is colored
# according to its assigned cluster label. The function uses matplotlib for
# plotting.

def plot_clusters(df, features):
    # Scatter plot with data points colored by cluster label
    plt.scatter(df[features[0]], df[features[1]], c=df['Cluster'], cmap='viridis')

    # Set x-axis label
    plt.xlabel(features[0])

    # Set y-axis label
    plt.ylabel(features[1])

    # Set plot title
    plt.title('Product Clusters')

    # Display the plot
    plt.show()

# The main function orchestrates the entire process, from loading and preprocessing
# data to training a GAN model, detecting anomalies (stock-outs), and visualizing
# product clusters. It serves as the entry point for the anomaly detection workflow.

def main():
    # Load data from the specified file path
    file_path = "/Users/igormol/Downloads/online_retail_II.csv"
    df = load_data(file_path)

    # Perform data preprocessing, focusing on the features 'Quantity' and 'Price'
    features = ['Quantity', 'Price']
    df = preprocess_data(df, features)

    # Apply k-means clustering to categorize products into clusters
    df = kmeans_clustering(df, features)

    # Label the identified clusters as 'Low', 'Average', or 'High-profitability'
    df = label_clusters(df)

    # Set hyperparameters for the GAN model
    latent_dim, output_dim, input_dim, epochs, batch_size = define_hyperparameters(features)

    # Build and compile the generator, discriminator, and GAN models
    generator = build_generator(latent_dim, output_dim)
    discriminator = build_discriminator(input_dim)
    discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')
    gan = build_gan(generator, discriminator, latent_dim)

    # Train the GAN model using the preprocessed data
    train_gan(generator, discriminator, gan, df, features, latent_dim, epochs, batch_size)

    # Detect anomalies (stock-outs) in the data using the trained GAN model
    stock_outs = anomaly_detection(df, features, generator, latent_dim)

    # Print the detected stock-outs
    print("Detected Stock-Outs:")
    print(stock_outs)

    # Visualize the product clusters
    plot_clusters(df, features)


if __name__ == "__main__":
    main()
