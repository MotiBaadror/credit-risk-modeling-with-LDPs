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


dataset = '../data/data_fs.csv'
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


# Map outputs to 0 (good) or 1 (bad)
# df = df.merge(pd.get_dummies(df.Risk, prefix='Risk'), left_index=True, right_index=True)
# del df['Risk']
# del df['Risk_good']
# df.to_csv('../data/features_data.csv', index=False)

# Separate X and y of dataset
# np.random.shuffle(df)
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


# K-fold cross validation that splits data into train and test set
# skf = StratifiedKFold(n_splits=5, shuffle=True) # default n_splits is 5

# Record highest mae and mse
top_mae = 0
top_mse = 0
# top_f1 = 0
# top_acc = 0
# top_bal_acc = 0

mean_fpr = np.linspace(0, 1, 100) # np.linspace returns 100 evenly spaced numbers over interval [0,1]
number_of_clusters = 23 # Why did they choose 23?
percentage_to_choose_from_each_cluster = 0.5 # Undersampling ratio of 0.5 for each majority class cluster

# y_test_all = []
# y_pred_all = []


for depth in range(2, 20, 10): # What is depth and estimators?
    for estimators in range(20, 50, 10):

        current_param_mae = []
        current_param_mse = []
        # current_param_f1 = []
        # current_param_acc = []
        # current_param_bal_acc = []
        current_param_aupr = []
        tprs = []

        for train_index, test_index in [np.split(np.arange(len(df)),[int(len(df)*.80)])]:
            # print('train_index:', train_index)
            # print('test_index:', test_index)

            X_train = X[train_index]
            X_test = X[test_index]
            # print('X_train:', X_train)
            # print('X_test:', X_test)
            
            y_train = y[train_index]
            y_test = y[test_index]
            # print('y_train:', y_train)
            # print('y_test:', y_test)
            
            # Cluster majority class instances
            # value, counts = np.unique(y_train, return_counts=True)
            # minority_class = value[np.argmin(counts)]
            # majority_class = value[np.argmax(counts)]

            # idx_min = np.where(y_train == minority_class)[0]
            # idx_maj = np.where(y_train == majority_class)[0]

            # majority_class_instances = X_train[idx_maj]
            # majority_class_labels = y_train[idx_maj]

            # kmeans = KMeans(n_clusters=number_of_clusters)
            # kmeans.fit(majority_class_instances)

            # X_maj = []
            # y_maj = []

            # points_under_each_cluster = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

            
            # Choose majority class instances and add to dataset to use
            # for key in points_under_each_cluster.keys():
            #     points_under_this_cluster = np.array(points_under_each_cluster[key])
            #     number_of_points_to_choose_from_this_cluster = math.ceil(
            #         len(points_under_this_cluster) * percentage_to_choose_from_each_cluster)
            #     selected_points = np.random.choice(points_under_this_cluster,
            #                                        size=number_of_points_to_choose_from_this_cluster)
            #     X_maj.extend(majority_class_instances[selected_points])
            #     y_maj.extend(majority_class_labels[selected_points])

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
            # y_test_all.extend(y_test)
            # y_pred_all.extend(y_pred)
            
            # print("y_test :", y_test)
            # print("predictions :", predictions)
            # print("y_pred :", y_pred)
            

            # Calculate mae and mse of current split with specified depth and estimators
            mae = mean_absolute_error(y_test, predictions) # predictions[:, 1] returns only second column (probability of bad)
            mse = mean_squared_error(y_test, predictions)
            # f1 = f1_score(y_test, y_pred)
            # acc = accuracy_score(y_test, y_pred)
            # bal_acc = balanced_accuracy_score(y_test, y_pred)
            # aupr = average_precision_score(y_test, predictions[:, 1])

            current_param_mae.append(mae)
            current_param_mse.append(mse)
            # current_param_mse.append(f1)
            # current_param_mse.append(acc)
            # current_param_mse.append(bal_acc)
            # current_param_aupr.append(aupr)

            # fpr, tpr, thresholds = roc_curve(y_test, predictions[:, 1])
            # tprs.append(interp(mean_fpr, fpr, tpr))
            # tprs[-1][0] = 0.0

        current_mean_mae = np.mean(np.array(current_param_mae))
        current_mean_mse = np.mean(np.array(current_param_mse))
        # current_mean_f1 = np.mean(np.array(current_param_f1))
        # current_mean_acc = np.mean(np.array(current_param_acc))
        # current_mean_bal_acc = np.mean(np.array(current_param_bal_acc))
        # current_mean_aupr = np.mean(np.array(current_param_aupr))

        
        # Compare new mae with current best
        if top_mae < current_mean_mae:
        # if top_mse < current_mean_mse:
            top_mae = current_mean_mae
            top_mse = current_mean_mse
            # top_f1 = current_mean_f1
            # top_acc = current_mean_acc
            # top_bal_acc = current_mean_bal_acc

            best_depth = depth
            best_estimators = estimators

            best_mae = top_mae
            best_mse = top_mse
            # best_f1 = top_f1
            # best_acc = top_acc
            # best_bal_acc = top_bal_acc
            # best_aupr = current_mean_aupr
            
            # print("top_mae :", top_mae)
            # print("best_mae :", best_mae)
            
            # best_tpr = np.mean(tprs, axis=0)
            # best_fpr = mean_fpr

            # best_precision, best_recall, _ = precision_recall_curve(y_test, predictions[:, 1])
            # best_fpr, best_tpr, thresholds = roc_curve(y_test, predictions[:, 1])


print('plotting', dataset)
# plt.clf()
# print("best_precision :", best_precision)
# print("best_recall :", best_recall)
print("best_depth :", best_depth)
print("best_estimators :", best_estimators)


# Precision-Recall and ROC curves
# plt.plot(best_recall, best_precision, lw=2, color='Blue',
#          label='Precision-Recall Curve')
# plt.plot(best_fpr, best_tpr, lw=2, color='red',
#          label='ROC curve')

# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# plt.legend(loc="upper right")
# plt.show()


# Evaluate model with other metrics
print("mae :", best_mae)                    # Higher is better
print("mse :", best_mse)                    # Closer to 1 is better
# print("F1 Score :", best_f1)                # Closer to 1 is better
# print("Accuracy :", best_acc)               # Closer to 1 is better
# print("Balanced Accuracy :", best_bal_acc)  # Best is 1, worst is 0

# plt.plot(fpr_c[1], tpr_c[1], lw=2, color='red',label='Roc curve: Clustered sampling') # Error: says fpr_c doesn't exist