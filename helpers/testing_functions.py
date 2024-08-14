# Pandas and numpy for data manipulation
import numpy as np
import pandas as pd
np.random.seed(42)


# Calculate the SSE for a Bayesian model
def calculate_sse(coefficients, X_test, y_test):

    var_means = pd.DataFrame(coefficients, index=[0])

    # Create an intercept column
    X_test_copy = X_test.copy()
    X_test_copy['intercept'] = 1

    # Align names of the test observations and means
    names = X_test_copy.columns[1:]
    X_test_copy = X_test_copy.loc[:, names]
    var_means = var_means[names]

    # Calculate estimate for each test observation using the average weights
    results = pd.DataFrame(index = X_test_copy.index, columns = ['estimate'])
    for row in X_test_copy.iterrows():
        results.loc[row[0], 'estimate'] = np.dot(np.array(var_means), np.array(row[1]))
        
    # Metrics 
    actual = np.array(y_test)
    errors = results['estimate'] - actual
    sse = np.mean(errors ** 2)

    return sse


# Calculate the binary comparison for the Bayesian model
# Take 2 samples from the posterior distribution and compare them to determine 
# if they are correctly ranked (i.e. if the first is larger than the second)
# Do this for 1000 random pairs of samples, and return the proportion of times
def calculate_binary_comparison(coefficients, X_test, y_test):

    var_means = pd.DataFrame(coefficients, index=[0])

    # Create an intercept column
    X_test_copy = X_test.copy()
    X_test_copy['intercept'] = 1

    # Align names of the test observations and means
    names = X_test_copy.columns[1:]
    X_test_copy = X_test_copy.loc[:, names]
    var_means = var_means[names]

    # Calculate estimate for each test observation using the average weights
    results = pd.DataFrame(index = X_test_copy.index, columns = ['estimate'])
    for row in X_test_copy.iterrows():
        results.loc[row[0], 'estimate'] = np.dot(np.array(var_means), np.array(row[1]))
        
    
    # Metrics 
    #actual = np.array(y_test)

    # Binary comparison
    total_correct = 0
    for i in range(1000):
        # Randomly select two samples from results
        sample1 = results.sample(1)
        sample2 = results.sample(1)

        # Make sure sample 1 and 2 are not the same sample
        while sample1.equals(sample2):
            sample2 = results.sample(1)

        # Get corresponding sample from actual
        sample1_actual = y_test[sample1.index[0]]
        sample2_actual = y_test[sample2.index[0]]

        # Determine which sample estimate is larger
        # If sample1 estimate is larger, and the same corresponding sample in y_test is larger, then the ranking is correct
        if(sample1['estimate'].values[0] > sample2['estimate'].values[0] and sample1_actual > sample2_actual):
            total_correct += 1
        elif(sample2['estimate'].values[0] > sample1['estimate'].values[0] and sample2_actual > sample1_actual):
            total_correct += 1
        elif(sample1['estimate'].values[0] == sample2['estimate'].values[0] and sample1_actual == sample2_actual):
            total_correct += 1
        else:
            total_correct += 0

    return total_correct / 1000



