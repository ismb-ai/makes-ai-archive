import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import multiprocessing

# Load the CSV file
file_path = "/Users/igormol/Desktop/RecSysDNA/sample/large_contextual_data.csv"
data = pd.read_csv(file_path)

# One-hot encode the categorical variables
contextual_variables = ['weather_condition', 'operating_system', 'device_type', 'age_group', 'gender', 'income_level', 'mood', 'activity', 'time_of_day', 'day_of_week', 'season']
data_encoded = pd.get_dummies(data, columns=contextual_variables)

# Function to create and display a table of feature importances for each product
def create_feature_importance_table(product_id, feature_importance_df):
    print(f"\nFeature Importances for Product ID {product_id}\n")
    print(feature_importance_df)

# Function to evaluate a subset of features
def evaluate(individual, X_train, y_train, X_test, y_test):
    # Extract the selected features
    selected_features = [feature for feature, included in zip(X_train.columns, individual) if included]
    if len(selected_features) == 0:
        return 0,

    # Train the logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train[selected_features], y_train)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test[selected_features])
    return accuracy_score(y_test, y_pred),

# Wrapper for parallel evaluation
def parallel_evaluate(individuals, toolbox):
    return Parallel(n_jobs=num_cores)(delayed(toolbox.evaluate)(ind) for ind in individuals)

# Use multiprocessing to speed up evaluations
num_cores = multiprocessing.cpu_count()

# DEAP setup
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", np.random.randint, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(data_encoded.columns) - 4)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Iterate over each product_id
for product_id in data['product_id'].unique():
    # Create a binary target variable indicating whether the product was sold or not
    data_encoded['sold'] = data['product_id'].apply(lambda x: 1 if x == product_id else 0)

    # Define the target variable
    y = data_encoded['sold']

    # The features are all columns except 'user_id', 'product_id', 'timestamp', and 'sold'
    X = data_encoded.drop(columns=['user_id', 'product_id', 'timestamp', 'sold'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Register the evaluation function
    toolbox.register("evaluate", evaluate, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    # Initialize the population
    population = toolbox.population(n=5)
    n_gen = 3
    halloffame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Set up the parallel map function
    toolbox.register("map", lambda f, x: parallel_evaluate(x, toolbox))

    # Run the genetic algorithm
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_gen, stats=stats, halloffame=halloffame, verbose=True)

    # Get the best individual (feature subset) from the genetic algorithm
    best_individual = halloffame[0]
    selected_features = [feature for feature, included in zip(X_train.columns, best_individual) if included]

    # Train the logistic regression model on the selected features
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train[selected_features], y_train)

    # Get the feature importances (coefficients)
    importance = model.coef_[0]

    # Create a DataFrame to store the feature names and their corresponding importances
    feature_importance_df = pd.DataFrame({
        'feature': selected_features,
        'importance': importance
    })

    # Sort the DataFrame by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    # Display the feature importance table for the current product
    create_feature_importance_table(product_id, feature_importance_df)
