import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

train = pd.read_csv('train.csv').iloc[:2000, :] 
X = train.drop(['Activity', 'subject'], axis=1).values
y = train['Activity'].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_labels = np.unique(y_encoded)

sc = StandardScaler()
X_std = sc.fit_transform(X)

n_features = X_std.shape[1]
mean_overall = np.mean(X_std, axis=0)
S_W = np.zeros((n_features, n_features))
S_B = np.zeros((n_features, n_features))

for c in class_labels:
    X_c = X_std[y_encoded == c]
    mean_c = np.mean(X_c, axis=0)
    
    sample_diff = (X_c - mean_c).T
    S_W += np.dot(sample_diff, sample_diff.T)
    
    n_c = X_c.shape[0]
    mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
    S_B += n_c * np.dot(mean_diff, mean_diff.T)

A = np.linalg.inv(S_W + np.identity(n_features) * 1e-6).dot(S_B)
eigenvalues, eigenvectors = np.linalg.eig(A)

idx = np.argsort(abs(eigenvalues))[::-1]
eigenvectors = eigenvectors.T[idx]

X_lda = X_std.dot(eigenvectors[0:2].T).real

plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_lda[:, 0], y=X_lda[:, 1], hue=y, palette='bright', alpha=0.6)
plt.show()