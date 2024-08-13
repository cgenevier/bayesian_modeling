# Pandas and numpy for data manipulation
import pandas as pd
import numpy as np
np.random.seed(42)

# Standard ML Models for comparison
from sklearn.linear_model import LinearRegression

# UCI ML Repo
from ucimlrepo import fetch_ucirepo

# Defines a class that will be used to load the dataset and model it
class Model:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.features = None
        self.total_data = None
        self.training_set_values = None
        self.ols_coefficients = None


    # Import the dataset from the UCI ML Repository
    def import_from_uci(self, uci_dataset_id, save_to_file=False):
        # Load the dataset
        training_df = fetch_ucirepo(id=uci_dataset_id)
        self.df = training_df.data.original

        # Save data and return it
        if(save_to_file):
            self.df.to_csv('uci_ml_datasets/ucirepo_' + str(uci_dataset_id) + '.csv', index=False)
        return self.df


    # Import the dataset from a file
    def import_from_file(self, file_path):
        self.df = pd.read_csv(file_path)
        return self.df
        

    # Normalize and format the data by setting the target column and splitting into features/target
    def format_data(self):
        # Normalize the data
        self.df = (self.df - self.df.mean())/self.df.std()

        # Set target to the last column in the dataframe
        target = self.df.columns[-1]
        self.df['target'] = self.df[target]
        self.df = self.df.drop(columns=[target])

        # sort by correlation with target (and maintain top X most correlated features with target if desired)
        # Find correlations with the target and maintain the top X most correlation features with target
        most_correlated = self.df.corr().abs()['target'].sort_values(ascending=False)
        #most_correlated = most_correlated[:8]
        self.df = self.df.loc[:, most_correlated.index]
            
        # Split df into features and target
        self.X = self.df.drop('target', axis=1)
        self.y = self.df['target']

        # Return features + target
        return self.X, self.y
    

    # Set up training dataset sizes 
    def set_up_training_values(self, training_set_sizes, max_training_set_percentage = 0.6):
        self.features = self.X.shape[1]
        self.total_data = self.X.shape[0]
        self.training_set_values = [x for x in training_set_sizes if x < max_training_set_percentage*self.total_data]
        return self.training_set_values
    

    # Determine signs of the coefficients using OLS
    def determine_signs_of_coefficients(self, ols_use_all_data=True, X_train=None, y_train=None):
        lr = LinearRegression()

        # Use Ordinary Least Squares Linear Regression to fit the data (training only or all data)
        if(ols_use_all_data):
            lr.fit(self.df, self.df['target'])
        else:
            lr.fit(X_train, y_train)

        self.ols_coefficients = {}
        ols_formula = 'target = %0.2f +' % lr.intercept_
        for i, col in enumerate(X_train.columns):
            ols_formula += ' %0.2f * %s +' % (lr.coef_[i], col)
            self.ols_coefficients[col] = lr.coef_[i]
            
        ' '.join(ols_formula.split(' ')[:-1])

        return self.ols_coefficients