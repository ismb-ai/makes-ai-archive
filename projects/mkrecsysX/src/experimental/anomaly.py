import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def generate_synthetic_data():
    np.random.seed(42)
    product_ids = np.random.randint(1, 101, 1000)
    client_ids = np.random.randint(1, 501, 1000)
    sales_dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)

    data = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
    data['ProductID'] = product_ids
    data['ClientID'] = client_ids
    data['SalesDate'] = sales_dates
    return data

def standardize_features(data):
    scaler = StandardScaler()
    data[['Feature1', 'Feature2']] = scaler.fit_transform(data[['Feature1', 'Feature2']])
    return data

def split_data(data):
    return train_test_split(data, test_size=0.2, random_state=42)

def build_generator():
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_dim=2))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(2, activation='linear'))
    return model

def build_discriminator():
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_dim=2))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def compile_discriminator(discriminator):
    discriminator.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

def create_combined_model(generator, discriminator):
    discriminator.trainable = False
    z = layers.Input(shape=(2,))
    gen_output = generator(z)
    dis_output = discriminator(gen_output)
    combined = models.Model(z, dis_output)
    combined.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    return combined

def train_gan(combined, discriminator, generator, X_train, epochs, batch_size):
    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_data = X_train.iloc[idx][['Feature1', 'Feature2']].values
        noise = np.random.normal(0, 1, (batch_size, 2))
        gen_data = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_data, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, 2))
        valid_y = np.ones((batch_size, 1))
        g_loss = combined.train_on_batch(noise, valid_y)

        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {d_loss[1]}] [G loss: {g_loss}]")

def detect_anomalies(generator, X_test):
    noise = np.random.normal(0, 1, (X_test.shape[0], 2))
    gen_data = generator.predict(noise)
    real_data = X_test[['Feature1', 'Feature2']].values
    anomalies = np.linalg.norm(real_data - gen_data, axis=1)
    X_test['AnomalyScore'] = anomalies
    return X_test

def print_anomalies(anomalies_df):
    print("Anomalies Detected:")
    print(anomalies_df)


def plot_anomalies(X_test):
    plt.scatter(X_test['Feature1'], X_test['Feature2'], c=X_test['AnomalyScore'], cmap='coolwarm')
    plt.colorbar(label='Anomaly Score')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Anomaly Detection')
    plt.show()

def main():
    data = generate_synthetic_data()
    data = standardize_features(data)
    X_train, X_test = split_data(data)

    generator = build_generator()
    discriminator = build_discriminator()

    compile_discriminator(discriminator)

    combined = create_combined_model(generator, discriminator)

    train_gan(combined, discriminator, generator, X_train, epochs=10, batch_size=64)

    anomalies_df = detect_anomalies(generator, X_test)

    print_anomalies(anomalies_df)

    plot_anomalies(X_test)

if __name__ == "__main__":
    main()
