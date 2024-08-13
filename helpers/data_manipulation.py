# Pandas and numpy for data manipulation
import numpy as np
np.random.seed(42)

# Standard ML Models for comparison
from sklearn.linear_model import LinearRegression


"""
Generate a k-by-N binary matrix for cross-validation indexing
    Parameters:
    k (int): Number of partitions (folds).
    N (int): Size of the current data set.
    training_set_value (int): Number of training set items to use

    Returns:
    np.ndarray: k-by-N binary matrix where 1 indicates a training set item and 0 indicates a test set item.
"""
def cv_indexing(k, N, training_set_value):
    
    indices = np.arange(N)
    np.random.shuffle(indices)
    
    # Distribute the remainder among the folds
    fold_sizes = np.full(k, training_set_value, dtype=int)
    fold_sizes[:training_set_value % k] += 1
    
    current = 0
    cv = np.zeros((k, N), dtype=int)  # Initialize with zeros for testing indicies
    
    for i in range(k):
        start, stop = current, current + fold_sizes[i]
        training_indices = indices[start:stop]
        
        # Mark training set items in cv
        cv[i, training_indices] = 1
        
        current = stop
    
    return cv

    

# Determine signs of the coefficients using OLS
def determine_signs_of_coefficients(df, ols_use_all_data=True, X_train=None, y_train=None):
    lr = LinearRegression()

    # Use Ordinary Least Squares Linear Regression to fit the data (training only or all data)
    if(ols_use_all_data):
        lr.fit(df, df['target'])
    else:
        lr.fit(X_train, y_train)

    ols_coefficients = {}
    ols_formula = 'target = %0.2f +' % lr.intercept_
    for i, col in enumerate(X_train.columns):
        ols_formula += ' %0.2f * %s +' % (lr.coef_[i], col)
        ols_coefficients[col] = lr.coef_[i]
        
    ' '.join(ols_formula.split(' ')[:-1])

    return ols_coefficients


        

# Normalize and format the data by setting the target column and splitting into features/target
def format_data(df):
    # Normalize the data
    df = (df - df.mean())/df.std()

    # Set target to the last column in the dataframe
    target = df.columns[-1]
    df['target'] = df[target]
    df = df.drop(columns=[target])

    print(df.head())

    # sort by correlation with target (and maintain top X most correlated features with target if desired)
    # Find correlations with the target and maintain the top X most correlation features with target
    most_correlated = df.corr().abs()['target'].sort_values(ascending=False)
    #most_correlated = most_correlated[:8]
    df = df.loc[:, most_correlated.index]
        
    # Split df into features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Return features + target
    return X, y, df