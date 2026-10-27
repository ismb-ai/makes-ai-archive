# Monte Carlo Anomaly Detector and Treatment for John Deere's Project
#
# Igor Mol <igor.mol@makes.ai>
#
# The provided Python code implements a Monte Carlo approach for detecting and 
# substituting outliers in a time series dataset. Monte Carlo methods are statistical 
# techniques that use random sampling to obtain numerical results. In this context, 
# the code leverages the Z-score method, a statistical measure that quantifies how far
# away a particular data point is from the mean of a dataset. By calculating the 
# Z-scores for each data point in the "COUNT" column of the time series, the code 
# identifies outliers—those points that deviate significantly from the mean. If 
# the Z-score surpasses a predefined threshold, the corresponding data point is 
# considered an outlier and is replaced with NaN. This approach allows for the 
# identification and treatment of extreme values in the time series data, enhancing
# the robustness of subsequent analyses.
# 	The primary motivation for employing Monte Carlo methods in outlier detection 
# lies in their ability to handle complex and uncertain systems. Monte Carlo techniques
# use random sampling, allowing for the exploration of a wide range of possible scenarios.
# In the context of outlier detection, this flexibility is advantageous when dealing 
# with diverse datasets and complex patterns that may be challenging to capture using
# deterministic methods alone. The Z-score-based outlier detection mechanism in the 
# code provides a statistically sound approach to identify and address outliers in time
# series data, which is crucial for ensuring the accuracy and reliability of subsequent
# analyses and modeling efforts.
# 	The code not only implements the outlier detection technique but also includes
# functions for generating synthetic time series data, plotting the original and cleaned 
# data on the same graph for visual comparison, and displaying the original and cleaned 
# datasets in well-formatted tables. These additional features contribute to the pedagogical
# value of the code by providing a comprehensive example of how to apply Monte Carlo 
# methods for outlier detection in a real-world scenario, while also emphasizing the
# importance of data visualization and presentation for effective communication of results
# in data science.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic data
def generate_synthetic_data():
    np.random.seed(42)
    data = pd.DataFrame({
        'TIME_SCALE': np.arange(1, 14),
        'COUNT': np.random.randint(10, 50, size=13)
    })
    return data

# Function to detect and substitute outliers using Z-score
def monte_carlo_outlier_detection(data_frame, column, threshold=3):
    z_scores = np.abs((data_frame[column] - data_frame[column].mean()) / data_frame[column].std())
    outliers = z_scores > threshold
    data_frame.loc[outliers, column] = np.nan
    return data_frame

# Plot original and cleaned data on the same plot
def plot_original_vs_cleaned(original_data, cleaned_data):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Plot original data
    sns.scatterplot(x='TIME_SCALE', y='COUNT', data=original_data, label='Original Data', color='skyblue')

    # Plot cleaned data
    sns.scatterplot(x='TIME_SCALE', y='COUNT', data=cleaned_data, label='Cleaned Data', color='salmon', marker='X')

    plt.title('Original vs. Cleaned Data')
    plt.xlabel('Time Scale')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

# Save the resulting time series in a data frame called "new_data"
def save_cleaned_data(original_data, column):
    new_data = original_data.copy()
    new_data = monte_carlo_outlier_detection(new_data, column)
    new_data.reset_index(drop=True, inplace=True)
    return new_data

# Display the data frames with improved formatting
def display_data_frames(original_data, cleaned_data):
    pd.set_option('display.precision', 2)
    
    print("\nOriginal Data:")
    print(original_data.to_string(index=False))

    print("\nCleaned Data:")
    print(cleaned_data.to_string(index=False))

# Main function to execute the steps
def main():
    original_data = generate_synthetic_data()
    cleaned_data = save_cleaned_data(original_data, 'COUNT')
    plot_original_vs_cleaned(original_data, cleaned_data)
    display_data_frames(original_data, cleaned_data)

if __name__ == "__main__":
    main()
