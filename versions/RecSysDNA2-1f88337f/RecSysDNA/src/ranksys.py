# ranksys.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

# This document contains the engine that performs the ranking of contextual
# variables (and their values) with respect to their relevance in determining
# whether product is sold for any given user. Our code implements a parallel
# computing approach (to enhance computational efficiency) using a Genetic
# Algorithm combined with a regression model. The result shows which features
# are the most likely in influencing sales' performance.

# In sec. 1, we collect functions to handle data and extract the predictions of
# the model. In sec. 2, we build the engine of our multi-core Genetic Algorithm.

import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
from deap import base, creator, tools, algorithms
from deap import base, creator, tools
import numpy as np

# Section 1: Data manipulation. Extraction of model's prediction.

def ranksys_one_hot_encode(data, contextual_variables):
    # Performs one-hot encoding on the given DataFrame 'data' using the contextual variables specified in 'contextual_variables'.
    # One-hot encoding converts categorical variables into a binary matrix representation, where each category becomes a new binary feature.
    return pd.get_dummies(data, columns=contextual_variables)

def ranksys_create_feature_importance_table(product_id, feature_importance_df):
    # Creates and displays a feature importance table for a given product ID.
    st.write(f"\nFeature Importances for Product ID {product_id}\n")
    st.dataframe(feature_importance_df)

def ranksys_get_feature_importances(model, selected_features):
    # Calculates feature importances based on the coefficients of a trained logistic regression model.
    # It takes the following parameters:
    # - 'model': A trained logistic regression model.
    # - 'selected_features': A list of selected features.

    # Extract the coefficients (importance) of features from the trained model.
    importance = model.coef_[0]

    # Create a DataFrame to store feature importances with their corresponding features.
    feature_importance_df = pd.DataFrame({
        'feature': selected_features,   # Features
        'importance': importance        # Corresponding importance (coefficient)
    })

    # Sort the DataFrame by feature importance in descending order.
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    # Return the DataFrame containing feature importances.
    return feature_importance_df

# Section 2: Engine

def ranksys_evaluate(individual, X_train, y_train, X_test, y_test):
    # Evaluates the performance of an individual in the genetic algorithm.
    # It takes the following parameters:
    # - 'individual': A binary vector representing the features selected for evaluation.
    # - 'X_train', 'y_train': Training data and labels.
    # - 'X_test', 'y_test': Test data and labels.

    # Extract the selected features based on the binary vector 'individual'.
    selected_features = [feature for feature, included in zip(X_train.columns, individual) if included]

    # If no features are selected, return a fitness score of 0.
    if len(selected_features) == 0:
        return 0,

    # Initialize a logistic regression model.
    model = LogisticRegression(max_iter=1000)

    # Fit the model on the training data using only the selected features.
    model.fit(X_train[selected_features], y_train)

    # Predict labels for the test data using the selected features.
    y_pred = model.predict(X_test[selected_features])

    # Calculate the accuracy score between the predicted and actual labels.
    accuracy = accuracy_score(y_test, y_pred)

    # Return the accuracy score as the fitness value.
    return accuracy,

def ranksys_parallel_evaluate(individuals, toolbox):
    # Performs parallel evaluation of a list of individuals using multiple CPU cores.
    # It takes the following parameters:
    # - 'individuals': A list of individuals (representing feature selection vectors) to evaluate.
    # - 'toolbox': The DEAP toolbox containing the evaluation function.

    # Get the number of CPU cores available.
    num_cores = multiprocessing.cpu_count()

    # Use Python's Parallel processing library to parallelize the evaluation of individuals.
    # The 'delayed' function is used to delay the evaluation of each individual until it's executed in parallel.
    # 'n_jobs' parameter specifies the number of CPU cores to use for parallel processing.
    # For each individual 'ind' in the list, evaluate it using the 'toolbox.evaluate' function.
    # Return the results of the parallel evaluations.
    return Parallel(n_jobs=num_cores)(delayed(toolbox.evaluate)(ind) for ind in individuals)

def create_fitness_and_individual():
    # Creates the DEAP FitnessMax and Individual classes if they don't already exist.

    # Check if the FitnessMax class exists in the creator module.
    if not hasattr(creator, "FitnessMax"):
        # If it doesn't exist, create it with base.Fitness as its base class and weights=(1.0,) as its default weights.
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    # Check if the Individual class exists in the creator module.
    if not hasattr(creator, "Individual"):
        # If it doesn't exist, create it as a list with fitness attribute set to FitnessMax.
        creator.create("Individual", list, fitness=creator.FitnessMax)

def register_attributes(toolbox, data_encoded):
    # Registers attributes and methods to the DEAP toolbox for use in the genetic algorithm.
    # It takes the following parameters:
    # - 'toolbox': The DEAP toolbox object to register attributes and methods.
    # - 'data_encoded': The encoded dataset used for initializing certain attributes.

    # Registering a random boolean attribute named 'attr_bool' using 'np.random.randint' function with a range of 2.
    toolbox.register("attr_bool", np.random.randint, 2)

    # Registering an individual initialization method named 'individual' using 'tools.initRepeat' function.
    # It initializes an individual with 'creator.Individual' class, using 'toolbox.attr_bool' for each gene,
    # and 'len(data_encoded.columns) - 4' as the number of genes.
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(data_encoded.columns) - 4)

    # Registering a population initialization method named 'population' using 'tools.initRepeat' function.
    # It initializes a population with a list of individuals, using 'toolbox.individual'.
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Registering a mating method named 'mate' using 'tools.cxTwoPoint'.
    # This method performs two-point crossover on individuals during the genetic algorithm.
    toolbox.register("mate", tools.cxTwoPoint)

    # Registering a mutation method named 'mutate' using 'tools.mutFlipBit'.
    # This method flips the bits of individuals with a probability of 'indpb=0.05' during the genetic algorithm.
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

    # Registering a selection method named 'select' using 'tools.selTournament'.
    # This method performs tournament selection during the genetic algorithm with a tournament size of 3.
    toolbox.register("select", tools.selTournament, tournsize=3)

def ranksys_initialize_deap(data_encoded):
    # Initializes the DEAP toolbox for a genetic algorithm.
    # It creates the necessary DEAP classes and registers attributes and methods using the 'register_attributes' function.
    # It takes the following parameter:
    # - 'data_encoded': The encoded dataset used for initializing certain attributes.

    # Create the DEAP FitnessMax and Individual classes if they don't already exist.
    create_fitness_and_individual()

    # Initialize the DEAP toolbox.
    toolbox = base.Toolbox()

    # Register attributes and methods to the toolbox using the 'register_attributes' function.
    register_attributes(toolbox, data_encoded)

    # Return the initialized toolbox.
    return toolbox

def ranksys_run_genetic_algorithm(toolbox, X_train, y_train, X_test, y_test):
    # Runs a genetic algorithm for feature selection using DEAP.
    # It takes the following parameters:
    # - 'toolbox': The DEAP toolbox containing the necessary functions for the genetic algorithm.
    # - 'X_train', 'y_train': Training data and labels.
    # - 'X_test', 'y_test': Test data and labels.

    # Register the evaluation function to the toolbox.
    # The evaluation function 'ranksys_evaluate' will be used to evaluate the fitness of individuals.
    # It is passed with training and testing data as additional arguments.
    toolbox.register("evaluate", ranksys_evaluate, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    # Define statistics for tracking during the genetic algorithm.
    stats = {
        "avg": np.mean,
        "std": np.std,
        "min": np.min,
        "max": np.max
    }

    # Create a Statistics object to collect statistics on the population's fitness values.
    # 'lambda ind: ind.fitness.values' extracts fitness values from individuals.
    stats_obj = tools.Statistics(lambda ind: ind.fitness.values)

    # Register each statistic with the Statistics object.
    for stat_name, stat_func in stats.items():
        stats_obj.register(stat_name, stat_func)

    # Configure parameters for the genetic algorithm.
    algorithm_params = {
        "cxpb": 0.5,  # Crossover probability
        "mutpb": 0.2,  # Mutation probability
        "ngen": 3,     # Number of generations
        "stats": stats_obj,                 # Statistics object for tracking
        "halloffame": tools.HallOfFame(1),  # Hall of Fame to keep the best individual
        "verbose": True                     # Enable verbose output
    }

    # Register parallel evaluation to speed up fitness evaluation using multiple CPU cores.
    # The 'map' method will be replaced with parallel evaluation.
    toolbox.register("map", lambda f, x: ranksys_parallel_evaluate(x, toolbox))

    # Initialize a population of individuals with a size of 5.
    population = toolbox.population(n=5)

    # Run the evolutionary algorithm using the simple evolutionary algorithm (eaSimple) method.
    # It evolves the population for a specified number of generations with the given parameters.
    algorithms.eaSimple(population, toolbox, **algorithm_params)

    # Return the best individual found by the genetic algorithm (the Hall of Fame).
    return algorithm_params["halloffame"]

def ranksys_train_model(X_train, y_train, selected_features):
    # Trains a logistic regression model using the selected features.
    # It takes the following parameters:
    # - 'X_train', 'y_train': Training data and labels.
    # - 'selected_features': A list of selected features used for training the model.

    # Initialize a logistic regression model with a maximum of 1000 iterations.
    model = LogisticRegression(max_iter=1000)

    # Fit the model on the training data using only the selected features.
    model.fit(X_train[selected_features], y_train)

    # Return the trained model.
    return model
