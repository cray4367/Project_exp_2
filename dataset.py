import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


n_samples = 5000
n_features = 5
false_data_ratio = 0.15


np.random.seed(42)


legit_data = np.random.normal(loc=0, scale=1, size=(n_samples, n_features))
legit_labels = np.zeros(n_samples)


n_false = int(n_samples * false_data_ratio)
false_data = legit_data[:n_false] + np.random.normal(0, 2, (n_false, n_features))
false_labels = np.ones(n_false)


X = np.vstack((legit_data, false_data))
y = np.hstack((legit_labels, false_labels))


indices = np.arange(len(X))
np.random.shuffle(indices)

X = X[indices]
y = y[indices]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


train_df = pd.DataFrame(X_train, columns=[f"Feature_{i+1}" for i in range(n_features)])
train_df['Label'] = y_train
train_df.to_csv('train.csv', index=False)

test_df = pd.DataFrame(X_test, columns=[f"Feature_{i+1}" for i in range(n_features)])
test_df['Label'] = y_test
test_df.to_csv('test.csv', index=False)

print(" Dataset generated successfully!")
print(" Train set: train.csv")
print(" Test set: test.csv")
