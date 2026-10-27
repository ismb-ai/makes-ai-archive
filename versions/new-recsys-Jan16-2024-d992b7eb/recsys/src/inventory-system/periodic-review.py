# Periodic Review System for Inventory Management

# Igor Mol <igor.mol@makes.ai>

import pandas as pd
import numpy as np
from datetime import datetime
from tabulate import tabulate  # Ensure you have tabulate installed

# Calculate Economic Order Quantity (EOQ)
def calculate_eoq(row):
    demand = row['Demand']
    average_costs = row['Average_Costs']
    
    # Assuming holding cost is half of the average cost
    holding_cost_per_unit = average_costs / 2  
    
    # Assuming ordering cost is half of the average cost
    ordering_cost_per_order = average_costs / 2  
    
    # Check if holding_cost_per_unit is zero to avoid division by zero
    if holding_cost_per_unit == 0:
        return 0
    
    # Calculate EOQ using the Economic Order Quantity formula
    eoq = np.sqrt((2 * demand * ordering_cost_per_order) / holding_cost_per_unit)
    return eoq

# Calculate Reorder Point (s)
def calculate_reorder_point(row):
    demand = row['Demand']
    lead_time = row['Lead_Time']
    
    # Assuming a service level of 95%
    z_score = 1.645
    
    # Calculate Reorder Point (s)
    reorder_point = int(demand * lead_time + z_score * np.sqrt(demand * lead_time))
    
    return reorder_point

# Load inventory DataFrame from CSV file
def load_inventory_data(file_path):
    inventory_df = pd.read_csv(file_path)
    inventory_df['Inclusion_Date'] = pd.to_datetime(inventory_df['Inclusion_Date'], format='%Y-%m-%d %H:%M:%S')
    
    # Calculate Lead Time
    current_date = datetime.now()
    inventory_df['Lead_Time'] = (current_date - inventory_df['Inclusion_Date']).dt.total_seconds() / (60 * 60 * 24)  # Convert seconds to days
    
    return inventory_df

# Filter products based on brand and Reorder_Point
def filter_products(data_frame, brand_keyword, max_reorder_point):
    filtered_table = data_frame[~data_frame['Product_Brand'].str.contains(brand_keyword, case=False) & (data_frame['Reorder_Point'] <= max_reorder_point)]
    return filtered_table[['Product_ID', 'Product_Name', 'Product_Brand', 'Reorder_Point']]

# Save the filtered result as a text file
def save_result_as_text(data_frame, output_file_path):
    data_frame.to_string(index=False, header=True, justify='center', buf=output_file_path)

# Main routine
def main():
    # Specify the file path
    file_path = "/Volumes/Untitled/cpp-inventory-new.csv"
    
    # Load inventory DataFrame from CSV file
    inventory_df = load_inventory_data(file_path)

    # Apply the functions to calculate EOQ and Reorder Point (s) for each product in the inventory_df
    inventory_df['EOQ'] = inventory_df.apply(calculate_eoq, axis=1)
    inventory_df['Reorder_Point'] = inventory_df.apply(calculate_reorder_point, axis=1)

    # Filter products based on brand and Reorder_Point
    filtered_table = filter_products(inventory_df, "DEERE", 50000)

    # Display the filtered table without Order Quantity
    print(tabulate(filtered_table, headers='keys', tablefmt='pretty'))

    # Save the filtered result as a text file
    output_file_path = "/Volumes/Untitled/cpp-inventory-analysis.txt"
    save_result_as_text(filtered_table, output_file_path)
    print(f"Filtered result saved to: {output_file_path}")

if __name__ == "__main__":
    main()
