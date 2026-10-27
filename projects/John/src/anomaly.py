# A Z-Score Approach to Anomaly Detection and Data Cleaning in Time Series
#
# Igor Mol <igor.mol@makes.ai>
#
# This code performs outlier detection and cleaning in time series data utilizing
# the Z-score method. The program defines a function to identify outliers based 
# on Z-scores and subsequently replaces them with NaN values. The cleaned time 
# series is then compared visually to the original data through a plot. The 
# resulting cleaned data is saved in a new DataFrame, allowing for a clear 
# understanding of the impact of outlier removal on the time series. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to detect and substitute outliers using Z-score
def z_score_outlier_detection(data_frame, column, threshold=3):
    # Calculate Z-scores for the specified column
    z_scores = np.abs((data_frame[column] - data_frame[column].mean()) / data_frame[column].std())
    
    # Identify outliers based on the threshold
    outliers = z_scores > threshold
    
    # Replace outlier values with NaN
    data_frame.loc[outliers, column] = np.nan
    
    # Return the data frame with replaced values
    return data_frame

# Plot original and cleaned data
def plot_original_vs_cleaned(original_data, cleaned_data):
    # Plot the original and cleaned data
    plt.figure(figsize=(10, 6))
    plt.plot(original_data['TIME_SCALE'], original_data['COUNT'], label='Original Data', marker='o')
    plt.plot(cleaned_data['TIME_SCALE'], cleaned_data['COUNT'], label='Cleaned Data', marker='X')
    plt.title('Original vs. Cleaned Data')
    plt.xlabel('Time Scale')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

# Save the resulting time series in a data frame called "new_data"
def save_cleaned_data(original_data, column):
    # Create a copy of the original data
    new_data = original_data.copy()
    
    # Apply outlier detection and replacement using Z-score
    new_data = z_score_outlier_detection(new_data, column)
    
    # Reset index for consistency
    new_data.reset_index(drop=True, inplace=True)
    
    # Return the cleaned data frame
    return new_data

# Display the original and cleaned data
original_data = data.copy()  # Assumes 'data' is the original DataFrame
cleaned_data = save_cleaned_data(original_data, 'COUNT')
plot_original_vs_cleaned(original_data, cleaned_data)

# Save the cleaned data in a new_data data frame
new_data = cleaned_data.copy()

# Display the resulting time series
print("Original Data:")
print(original_data)
print("\nCleaned Data:")
print(new_data)
