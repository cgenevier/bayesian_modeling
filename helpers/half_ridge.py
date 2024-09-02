# Pandas and numpy for data manipulation
import numpy as np
import pandas as pd
np.random.seed(42)

# Warnings for warning suppression
import warnings

# PyMC for Bayesian Inference
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


# Use rejection sampling to fit a half-ridge regression model to the data
def half_ridge_rejection_sampling(weight_signs, ols_coefficients, X_train, y_train, prior_eta, chain_length):

    if(prior_eta == 0):
        # Create post_weights array with the same magnitude but with signs of ols_coefficients
        value = np.sqrt(2 / np.pi) # From 2018 Paper
        post_weights = weight_signs * value 
        post_weights = dict(zip(X_train.columns, post_weights))
        return post_weights
    elif(prior_eta == np.inf):
        # Simple OLS at the limit (with euclidean projection according to correct signs)

        # Keys are in a specific order that matches the order of weight_signs
        keys = list(ols_coefficients.keys())

        # Loop through the dictionary and compare the signs
        for i, key in enumerate(keys):
            correct_sign = np.sign(weight_signs[i])  # Get the corresponding correct sign from weight_signs
            current_sign = np.sign(ols_coefficients[key])
            
            # If the current sign doesn't match the correct sign, set the coefficient to 0
            if current_sign != correct_sign:
                ols_coefficients[key] = 0

        # NOTE: This may not be 100% correct - Matt says that 
        # this is a good approximation but not the exact solution
        # which requires the Mahalanobis distance I believe

        return ols_coefficients

    # assumption that cue directionalities are known in advance (Dawes, 1979)
    col_pos = (weight_signs > 0).astype(int)
    total_features = len(ols_coefficients)

    # Set penalty (lambda = sigma^2/eta^2) and sigma^2 (sigma = 1/eta)
    sigma_2 = 1/(prior_eta ** 2)
    penalty = 1/(prior_eta ** 4)

    # Convert pandas dataframe to arrays
    X = X_train.values
    Y = y_train.values
    
    ### L2 half-ridge Fitting (Training Data) ###

    # posterior mean is a vector that is total_features-dimensional, as a function of penalty
    post_mean = np.dot(pinv(penalty * np.eye(total_features) + np.dot(X.T, X)), np.dot(X.T, Y))
    
    # posterior variance is a total_features x total_features covariance square matrix
    post_var = pinv(sigma_2 * np.eye(total_features) + (1/sigma_2)* np.dot(X.T, X))
    
    # Ensure the posterior variance matrix is symmetric positive-definite
    post_var = nearest_posdef(post_var)

    # Sampling from a multivariate Gaussian with above mean and variance
    samples1 = multivariate_normal.rvs(mean=post_mean, cov=post_var, size=chain_length)

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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        post_weights = np.nanmean(samples1, axis=0)
    
    # If all samples are NaN, retry sampling up to 3 times
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

