
from surprise import Dataset, Reader, KNNBasic
import pandas as pd
from tabulate import tabulate

# Load the original DataFrame
csv_file_path = '/Users/igormol/Downloads/online_retail_II.csv'
df = pd.read_csv(csv_file_path)

# Sum the 'Quantity' values for each unique pair of 'Customer ID' and 'StockCode'
user_item_interaction = df.groupby(['Customer ID', 'StockCode'])['Quantity'].sum().reset_index()

# Create a Reader object to specify the rating scale
reader = Reader(rating_scale=(user_item_interaction['Quantity'].min(), user_item_interaction['Quantity'].max()))

# Load the data into the Surprise Dataset
data = Dataset.load_from_df(user_item_interaction[['Customer ID', 'StockCode', 'Quantity']], reader)

# Build the user-item interaction matrix
trainset = data.build_full_trainset()

# Initialize the k-NN collaborative filtering model with Pearson similarity
sim_options = {'name': 'pearson', 'user_based': True}
knn_model = KNNBasic(sim_options=sim_options)

try:
    # Train the model on the data
    knn_model.fit(trainset)
except Exception as e:
    print(f"Error during training: {e}")

# Get recommendations for all clients
all_customer_ids = user_item_interaction['Customer ID'].unique()

for customer_id in all_customer_ids:
    # Get items the customer has not yet interacted with
    items_not_interacted = user_item_interaction.loc[user_item_interaction['Customer ID'] != customer_id, 'StockCode'].unique()

    # Predict ratings for items not interacted
    predictions = []
    for item in items_not_interacted:
        try:
            prediction = knn_model.predict(customer_id, item)

            # Rescale the estimated quantity to [1, 5]
            min_quantity = user_item_interaction['Quantity'].min()
            max_quantity = user_item_interaction['Quantity'].max()
            normalized_score = 1 + 4 * ((prediction.est - min_quantity) / (max_quantity - min_quantity))

            predictions.append({'StockCode': item, 'Normalized_Score': normalized_score})
        except Exception as e:
            print(f"Error predicting for Customer ID {customer_id} and StockCode {item}: {e}")

    # Sort predictions by normalized score in descending order
    sorted_predictions = sorted(predictions, key=lambda x: x['Normalized_Score'], reverse=True)

    # Get the top N recommendations (e.g., top 5)
    top_recommendations = sorted_predictions[:5]

    # Organize the recommendations in a table
    table_data = []
    for rec in top_recommendations:
        table_data.append([rec['StockCode'], rec['Normalized_Score']])

    # Print top recommendations for the current customer as a table
    print(f"\nTop recommendations for Customer ID {customer_id}:\n")
    print(tabulate(table_data, headers=['StockCode', 'Normalized Score'], tablefmt='grid'))
