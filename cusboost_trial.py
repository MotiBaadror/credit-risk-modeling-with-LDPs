import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import interp
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import math
from sklearn.metrics import  average_precision_score, matthews_corrcoef, f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import seaborn as sns


dataset = 'data/data_fs.csv'
print("dataset : ", dataset)
df = pd.read_csv(dataset)

# Drop first column containing original row numbers
df.drop('Unnamed: 0', axis=1, inplace=True)
df.head()

print("Age: ", df['Age'].unique())
print("Sex: ", df['Sex'].unique())
print("Job: ", df['Job'].unique())
print("Housing: ", df['Housing'].unique())
print("Saving accounts: ", df['Saving accounts'].unique())
print("Checking account: ", df['Checking account'].unique())
# print("Credit amount: ", df['Credit amount'].unique())
# print("Duration: ", df['Duration'].unique())
print("Purpose: ", df['Purpose'].unique())
# print("Risk: ", df['Risk'].unique())


# One hot encoding function
def one_hot(df, nan = False):
    original = list(df.columns)
    category = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns = category, dummy_na = nan, drop_first = True)
    new_columns = [c for c in df.columns if c not in original]
    return df, new_columns

# Feature extraction
df = df.merge(pd.get_dummies(df['Sex'], drop_first=True, prefix='Sex'), left_index=True, right_index=True)
df = df.merge(pd.get_dummies(df['Housing'], drop_first=True, prefix='Housing'), left_index=True, right_index=True)
df = df.merge(pd.get_dummies(df["Saving accounts"], drop_first=False, prefix='Saving'), left_index=True, right_index=True)
df = df.merge(pd.get_dummies(df["Checking account"], drop_first=False, prefix='Checking'), left_index=True, right_index=True)
df = df.merge(pd.get_dummies(df['Purpose'], drop_first=True, prefix='Purpose'), left_index=True, right_index=True)

# Group age into categories
interval = (18, 25, 40, 65, 100)
categories = ['University', 'Younger', 'Older', 'Senior']
df["Age_cat"] = pd.cut(df.Age, interval, labels=categories)
df = df.merge(pd.get_dummies(df["Age_cat"], drop_first=True, prefix='Age_cat'), left_index=True, right_index=True)
# print("Age_cat: ", df['Age_cat'].unique())

# Delete old columns
del df['Sex']
del df['Housing']
del df['Saving accounts']
del df['Checking account']
del df['Purpose']
del df['Age']
del df['Age_cat']

# Scale credit amount by natural log function
df['Credit amount'] = np.log(df['Credit amount'])



# Separate X and y of dataset
X = np.array(df.drop(['Credit amount'], axis=1))
y = np.array(df['Credit amount'])
shuffle(X,y)
print("X:", X, '\n')
# print("y:", y, '\n')

# Rescale feature values to decimals between 0 and 1
normalization_object = Normalizer()
# X_norm = normalization_object.fit_transform(X)
# X = X_norm
# print("X_norm:", X_norm)



# Record highest mae and mse
top_mae = 0
top_mse = 0



for depth in range(2, 20, 10): # What is depth and estimators?
    for estimators in range(20, 50, 10):

        current_param_mae = []
        current_param_mse = []

        for train_index, test_index in [np.split(np.arange(len(df)),[int(len(df)*.80)])]:

            X_train = X[train_index]
            X_test = X[test_index]
            
            y_train = y[train_index]
            y_test = y[test_index]
            # Cluster majority class instances
            X_sampled = X_train # np.concatenate((X_train[idx_min], np.array(X_maj)))
            y_sampled = y_train #np.concatenate((y_train[idx_min], np.array(y_maj)))

            # Use AdaBoost as ensemble regressor of Decision Trees
            regressor = AdaBoostRegressor(
                DecisionTreeRegressor(max_depth=depth),
                n_estimators=estimators,
                learning_rate=0.001) # SAMME discrete boosting algorithm, SAMME.R real boosting algorithm (converges faster)
            
            
            # Train regressor
            regressor.fit(X_sampled, y_sampled)
            # print("Trained regressor :", regressor.fit(X_sampled, y_sampled))
            
            # Make predictions on test data
            predictions = regressor.predict(X_test) # Nx2 array of probabilities in class 0 (good) and class 1 (bad) where N is 1000/(# splits)
            y_pred = regressor.predict(X_test) # Returns N predicted y and agrees with probabilities
            

            # Calculate mae and mse of current split with specified depth and estimators
            mae = mean_absolute_error(y_test, predictions) # predictions[:, 1] returns only second column (probability of bad)
            mse = mean_squared_error(y_test, predictions)

            current_param_mae.append(mae)
            current_param_mse.append(mse)

        current_mean_mae = np.mean(np.array(current_param_mae))
        current_mean_mse = np.mean(np.array(current_param_mse))

        
        # Compare new mae with current best
        if top_mae < current_mean_mae:
        # if top_mse < current_mean_mse:
            top_mae = current_mean_mae
            top_mse = current_mean_mse

            best_depth = depth
            best_estimators = estimators

            best_mae = top_mae
            best_mse = top_mse


print('plotting', dataset)
# plt.clf()
print("best_mae :", best_mae)
print("best_mse :", best_mse)
print("best_depth :", best_depth)
print("best_estimators :", best_estimators)




# Evaluate model with other metrics
print("mae :", best_mae)                    # Higher is better
print("mse :", best_mse)                    # Closer to 1 is better
