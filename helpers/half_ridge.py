# Pandas and numpy for data manipulation
import numpy as np
import pandas as pd
np.random.seed(42)

# PyMC for Bayesian Inference
import arviz as az
import pymc as pm

# Import scipy stats
from scipy.linalg import pinv, eigh
from scipy.stats import multivariate_normal

# Use PYMC to fit a half-ridge regression model to the data
def half_ridge_mcmc(X_train, y_train, ols_coefficients, prior_eta=100):

    # Define the variables needed for the model (uses R notation)
    formula = 'target ~ ' + ' + '.join(['%s' % variable for variable in X_train.columns])
    target, predictors = formula.split(' ~ ')

    with pm.Model() as model:  

        # Define prior intercept (truncated gaussian distributions with varied tau)
        intercept = pm.TruncatedNormal('intercept', mu=0, sigma=prior_eta, lower=0)

        # Define the priors for the weights (truncated gaussian distributions with varied tau)
        weights = []
        for predictor in predictors.split(' + '):
            if(ols_coefficients[predictor] >= 0):
                weights.append(pm.TruncatedNormal(predictor, mu=0, sigma=prior_eta, lower=0))
            else:
                weights.append(pm.TruncatedNormal(predictor, mu=0, sigma=prior_eta, upper=0))

        # Define the likelihood (normal distribution with a mean that is a linear combination of the weights and predictors, and tau fixed at 1)
        mu = intercept
        for i, predictor in enumerate(predictors.split(' + ')):
            mu += weights[i] * X_train[predictor]
        likelihood = pm.Normal("y", mu=mu, sigma=(1/prior_eta), observed=y_train)

        # @todo: See if there's a closed form way of determining the mean and variance of the truncated normal distribution rather than using MCMC
        # Perform Markov Chain Monte Carlo sampling
        #normal_trace = pm.sample(draws=2000, chains=4, tune=1000, step=pm.Metropolis(tune=1000))

        # Hyperparameters to tune?
        #1. Learning Rate: This controls the step size during the optimization process.
	    #2.	Number of Iterations/Samples: The number of iterations or samples to draw from the posterior distribution.
	    #3.	Burn-in Period: The number of initial samples to discard to allow the model to reach a stable state.
	    #4.	Thinning: The interval at which samples are retained to reduce autocorrelation.

        # draw 3000 posterior samples using NUTS sampling
        normal_trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.9)
    
    return normal_trace


# Calculate Bayesian posterior deterministically for gaussian likelihood and prior (effectively ridge regression)
def ridge_regression(X_train, y_train):
    # Prior variance (eta^2)
    eta2 = 1.0

    # Likelihood variance (sigma^2)
    sigma2 = 1.0

    # Compute posterior covariance
    XtX = np.dot(X_train.T, X_train)
    precision_prior = np.eye(X_train.shape[1]) / eta2
    precision_likelihood = XtX / sigma2
    precision_post = precision_likelihood + precision_prior
    cov_post = np.linalg.inv(precision_post)

    # Compute posterior mean
    XtY = np.dot(X_train.T, y_train)
    mean_post = np.dot(cov_post, XtY) / sigma2

    print("Posterior Mean:\n", mean_post)
    print("Posterior Covariance:\n", cov_post)

    return mean_post, cov_post


# Use rejection sampling to fit a half-ridge regression model to the data
def half_ridge_rejection_sampling(ols_coefficients, X_train, y_train, prior_eta, chain_length):
    print('Running half-ridge regression...')

    if(prior_eta == 0):
        # Create post_weights array with the same magnitude but with signs of ols_coefficients
        value = np.sqrt(2 / np.pi) # From 2018 Paper
        post_weights = np.sign(list(ols_coefficients.values())) * value 
        post_weights = dict(zip(X_train.columns, post_weights))
        print(post_weights)
        return post_weights
    elif(prior_eta == np.inf):
        # Simple OLS at the limit
        return ols_coefficients


    # assumption that cue directionalities are known in advance (Dawes, 1979)
    unitweights = np.sign(list(ols_coefficients.values()))
    col_pos = (unitweights > 0).astype(int)
    total_features = len(ols_coefficients)

    # Set penalty (lambda = sigma^2/eta^2) and sigma^2 (sigma = 1/eta)
    sigma_2 = 1/(prior_eta ** 2)
    penalty = 1/(prior_eta ** 4)

    # Convert pandas dataframe to arrays
    X = X_train.values
    Y = y_train.values
    
    ### L2 half-ridge Fitting (Training Data) ###
    inter = pinv(np.dot(X.T, X) + sigma_2 * penalty * np.eye(total_features))

    # posterior mean is a vector that is total_features-dimensional, as a function of penalty
    post_mean = np.dot(inter, np.dot(X.T, Y))
    
    # posterior variance is a total_features x total_features covariance square matrix
    post_var = sigma_2 * inter
    
    # Ensure the posterior variance matrix is positive-definite
    if not is_positive_definite(post_var):
        post_var = nearest_posdef(post_var)
    
    # Sampling from a multivariate Gaussian with above mean and variance
    samples1 = multivariate_normal.rvs(mean=post_mean, cov=post_var, size=chain_length)
    
    # Ensure samples1 is 2D
    if chain_length == 1:
        samples1 = samples1[np.newaxis, :]
    
    # Cut-off rule: reflects the correct cue directions that are known in advance (ols_coefficients)
    # Find rows with features that have an incorrect sign
    #incorrect_signs = np.any((samples1[:, col_pos == 1] < 0) | (samples1[:, col_pos == 0] > 0), axis=1)

    # Identify the column indices for positive and negative weights
    positive_indices = np.where(col_pos == 1)[0]
    negative_indices = np.where(col_pos == 0)[0]

    # Cut-off rule: reflects the correct cue directions that are known in advance (ols_coefficients)
    # Find rows with features that have an incorrect sign
    incorrect_positive = np.any(samples1[:, positive_indices] < 0, axis=1)
    incorrect_negative = np.any(samples1[:, negative_indices] > 0, axis=1)
    incorrect_signs = incorrect_positive | incorrect_negative


    # Set the entire row to NaN where any incorrect value is found
    samples1[incorrect_signs, :] = np.nan
    
    # Calculate the posterior weights
    post_weights = np.nanmean(samples1, axis=0)
    
    # If all samples are NaN, retry sampling
    i = 0
    while np.isnan(post_weights).any() and i < 3:
        i += 1
        samples1 = multivariate_normal.rvs(mean=post_mean, cov=post_var, size=50000)
        samples1[:, col_pos == 1][samples1[:, col_pos == 1] < 0] = np.nan
        samples1[:, col_pos == 0][samples1[:, col_pos == 0] > 0] = np.nan
        post_weights = np.nanmean(samples1, axis=0)
    
    if np.isnan(post_weights).any():
        raise ValueError('NA error in posterior weights')
    
    # Convert post_weights to a dictionary with appropriate column names
    post_weights = dict(zip(X_train.columns, post_weights))
    
    return post_weights


# Find the nearest positive-definite matrix to a given matrix
def nearest_posdef(A):
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if is_positive_definite(A3):
        return A3
    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not is_positive_definite(A3):
        mineig = np.min(np.real(eigh(A3)[0]))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
    return A3


# Check if a matrix is positive definite
def is_positive_definite(x):
    return np.all(np.linalg.eigvals(x) > 0)


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

        if i==20:
            print('Samples')
            print(total_correct)
            print(sample1)
            print(sample2)
            print(sample1_actual)
            print(sample2_actual)

    return total_correct / 1000





