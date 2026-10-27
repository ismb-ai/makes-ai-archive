import csv
import random
from datetime import datetime, timedelta

def generate_client_ids(num_clients):
    # Generate random client IDs
    return [random.randint(1000, 9999) for _ in range(num_clients)]

def generate_sales_data(client_ids, num_lines):
    # Generate random sales data with assigned client IDs
    sales_data = []
    for _ in range(num_lines):
        client_id = random.choice(client_ids)
        product_id = random.randint(100, 999)
        sale_value = round(random.uniform(10, 1000), 2)
        sale_date = datetime.now() - timedelta(days=random.randint(1, 365))
        quantity = random.randint(1, 10)
        sales_data.append([client_id, product_id, sale_value, sale_date.strftime('%Y-%m-%d'), quantity])
    return sales_data

def save_to_csv(data, filename):
    # Save data to CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Client_ID', 'Product_ID', 'Sale_Value', 'Sale_Date', 'Quantity'])
        writer.writerows(data)

def main():
    num_clients = 50  # Number of clients
    client_ids = generate_client_ids(num_clients)
    
    num_lines = 1000  # Number of lines in the CSV file
    sales_data = generate_sales_data(client_ids, num_lines)
    save_to_csv(sales_data, 'sales_inventory_template.csv')
    print(f"CSV file 'sales_inventory_template.csv' generated with {num_lines} lines.")

if __name__ == "__main__":
    main()

