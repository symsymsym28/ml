import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('mushroom.csv')
le = LabelEncoder()
for column in dataset.columns:
    dataset[column] = le.fit_transform(dataset[column])

X = dataset.drop('class', axis=1).values
y = dataset['class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

def plot_results(cm, accuracy):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.subplot(1, 2, 2)
    plt.bar(['Accuracy'], [accuracy], color="green")
    plt.ylim(0, 1)
    plt.show()

plot_results(cm, accuracy)