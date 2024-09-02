# Pandas and numpy for data manipulation
import pandas as pd
import numpy as np
np.random.seed(42)

# Pairwise combinations
from itertools import combinations

# Label encoding
from sklearn.preprocessing import LabelEncoder

# Standard ML Models for comparison
from sklearn.linear_model import LinearRegression

# UCI ML Repo
from ucimlrepo import fetch_ucirepo

# Defines a class that will be used to load the dataset and model it
class Model:
    def __init__(self):
        self.dataset_info = None
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

        # Iterate over number of targets (some UCI datasets have more than one) and save one dataset per target:
        for i in range(training_df.data.targets.shape[1]):
            # Get the name of the target column
            target_column = training_df.data.targets.iloc[:, i].name

            # Remove any columns in features that match the target_column name
            features_df = training_df.data.features.drop(columns=[target_column], errors='ignore')

            # Combine the modified features and the target column
            self.df = pd.concat([features_df, training_df.data.targets.iloc[:, i]], axis=1)

            # Remove '%' Signs
            for column in self.df.columns:
                # Check if all values in the column end with '%' and are strings
                if self.df[column].apply(lambda x: isinstance(x, str) and x.endswith('%')).all():
                    # Remove the '%' symbol and convert the column to numeric
                    self.df[column] = self.df[column].str.rstrip('%').astype(float)

            # Save data and return it
            if(save_to_file):
                self.df.to_csv('uci_ml_datasets/ucirepo_' + str(uci_dataset_id) + '_' + str(i) + '.csv', index=False)
                
        return self.df


    # Import the dataset from a file
    def import_from_file(self, file_path):
        self.df = pd.read_csv(file_path)
        return self.df
    
    
    # Get dataset metadata
    def get_dataset_info(self, uci_dataset_id):
        # Load the dataset
        training_df = fetch_ucirepo(id=uci_dataset_id)

        # Get the dataset metadata
        self.dataset_info = training_df.metadata

        return self.dataset_info

        

    # Normalize and format the data by setting the target column and splitting into features/target
    def format_data(self, categorical_encoding, missing_data_handling):

        # Handle missing values (except for drop, numerical values only)
        if missing_data_handling == 'drop':
            self.df = self.df.dropna()
        elif missing_data_handling == 'mean':
            self.df[self.df.select_dtypes(include=['number']).columns] = self.df.select_dtypes(include=['number']).apply(lambda x: x.fillna(x.mean()))
        elif missing_data_handling == 'median':
            self.df[self.df.select_dtypes(include=['number']).columns] = self.df.select_dtypes(include=['number']).apply(lambda x: x.fillna(x.median()))
        elif missing_data_handling == 'mode':
            self.df[self.df.select_dtypes(include=['number']).columns] = self.df.select_dtypes(include=['number']).apply(lambda x: x.fillna(self.df.mode().iloc[0]))
        elif missing_data_handling == 'zero':
            self.df[self.df.select_dtypes(include=['number']).columns] = self.df.select_dtypes(include=['number']).apply(lambda x: x.fillna(0))
        elif missing_data_handling == 'ffill':
            self.df[self.df.select_dtypes(include=['number']).columns] = self.df.select_dtypes(include=['number']).apply(lambda x: x.fillna(method='ffill'))
        elif missing_data_handling == 'bfill':
            self.df[self.df.select_dtypes(include=['number']).columns] = self.df.select_dtypes(include=['number']).apply(lambda x: x.fillna(method='bfill'))
        elif missing_data_handling == 'interpolate':
            self.df[self.df.select_dtypes(include=['number']).columns] = self.df.select_dtypes(include=['number']).apply(lambda x: x.interpolate())

        # Adjust categorical data 
        if categorical_encoding == 'label': # use label encoding
            # Identify categorical columns
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns

            # Initialize LabelEncoder
            label_encoder = LabelEncoder()

            # Apply LabelEncoder to each categorical column
            for col in categorical_cols:
                self.df[col] = label_encoder.fit_transform(self.df[col])

        elif categorical_encoding == 'onehot': # use one-hot encoding, drop the categorical column
            self.df = pd.get_dummies(self.df, drop_first=True)

        # Normalize the data
        #self.df = (self.df - self.df.mean())/self.df.std()
        self.df = (self.df - self.df.mean()) / self.df.std().where(self.df.std() != 0, 1)

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

        

    # Create paired data for binary comparison
    def create_paired_data(self):
        N = len(self.X)
        col_cues = self.X.columns
        
        # Generate all 2-element combinations of row indices
        comb = np.array(list(combinations(range(N), 2)))
        
        # Shuffle each pair randomly
        comb = np.apply_along_axis(np.random.permutation, 1, comb)
        
        y_pairs = np.zeros(comb.shape[0])
        bdata_diff = np.zeros((comb.shape[0], len(col_cues)))

        comparisons = []

        for i, (index1, index2) in enumerate(comb):
            # Extract the two rows based on indices
            binary = self.X.iloc[[index1, index2]].values
            y1, y2 = self.y.iloc[[index1, index2]]

            # Store the binary comparison data
            comparisons.append(self.X.iloc[[index1, index2]])

            # Compare y values and assign 1 or -1
            if y1 > y2:
                y_pairs[i] = 1  # A is better
            else:
                y_pairs[i] = -1  # B is better

            # Calculate the difference in feature values
            bdata_diff[i, :] = binary[0] - binary[1]

        # Convert differences to DataFrame and add labels
        bdata_diff = pd.DataFrame(bdata_diff, columns=col_cues)
        
        # Convert y_pairs to a Series
        y_pairs = pd.Series(y_pairs)
    
        # Update self.X and self.y with paired data
        self.X = bdata_diff
        self.y = y_pairs
    

    # Set up training dataset sizes 
    def set_up_training_values(self, training_set_sizes, max_training_set_percentage = 0.6):
        self.features = self.X.shape[1]
        self.total_data = self.X.shape[0]
        self.training_set_values = [x for x in training_set_sizes if x < max_training_set_percentage*self.total_data]
        return self.training_set_values
    

    # Determine signs of the coefficients using OLS
    def get_ols_coefficients(self, ols_use_all_data=True, X_train=None, y_train=None):
        lr = LinearRegression()

        # Use Ordinary Least Squares Linear Regression to fit the data (training only or all data)
        if(ols_use_all_data):
            lr.fit(self.X, self.y)
            columns = self.X.columns
        else:
            lr.fit(X_train, y_train)
            columns = X_train.columns

        self.ols_coefficients = {}
        for i, col in enumerate(columns):
            self.ols_coefficients[col] = lr.coef_[i]

        return self.ols_coefficients
    
    """
    Generate a k-by-N binary matrix for cross-validation indexing
        Parameters:
        k (int): Number of partitions (folds).
        training_set_value (int): Number of training set items to use

        Returns:
        np.ndarray: k-by-N binary matrix where 1 indicates a training set item and 0 indicates a test set item.
    """
    def cv_indexing(self, k, training_set_value):
        N = self.total_data
        
        indices = np.arange(N)
        np.random.shuffle(indices)
        
        current = 0
        cv = np.zeros((k, N), dtype=int)  # Initialize with zeros for testing indices

        for i in range(k):
            start, stop = current, current + training_set_value
            training_indices = indices[start:stop]

            # Handle wrap-around
            if start > N:
                # set training_indices equal to indexes from start % N to stop % N
                training_indices = indices[start % N:stop % N]
            elif stop > N:
                training_indices = np.concatenate((indices[start % N:], indices[:stop % N]))
            else:
                training_indices = indices[start:stop]

            # Mark training set items in cv
            cv[i, training_indices] = 1

            current = stop % N
            
        return cv