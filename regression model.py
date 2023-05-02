from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


iris = load_iris()


X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Building logistic regression model 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


lr = LogisticRegression()
lr.fit(X_train, y_train)


y_pred_lr = lr.predict(X_test)

# Compute the accuracy
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {acc_lr:.2f}")

#ANN CODE 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score


y_train_nn = to_categorical(y_train)
y_test_nn = to_categorical(y_test)


model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train_nn, epochs=100, verbose=0)


y_pred_nn = model.predict_classes(X_test)


acc_nn = accuracy_score(y_test, y_pred_nn)
print(f"Artificial Neural Network Accuracy: {acc_nn:.2f}")

#Performance matrices 
from sklearn.metrics import classification_report


print("Logistic Regression Metrics:")
print(classification_report(y_test, y_pred_lr))


print("Artificial Neural Network Metrics:")
print(classification_report(y_test, y_pred_nn))

# Plots for regression model and ANN 

import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions


X = np.vstack((X_train, X_test))
y = np.hstack((y_train, y_test))


plt.figure(figsize=(8, 6))
plot_decision_regions(X, y, clf=lr, legend=2)
plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Sepal Length (Standardized)")
plt.ylabel("Sepal Width (Standardized)")
plt.show()


plt.figure(figsize=(8, 6))
plot_decision_regions(X, y, clf=model, legend=2)
plt.title("Artificial Neural Network Decision Boundary")
plt.xlabel("Sepal Length (Standardized)")
plt.ylabel("Sepal Width (Standardized)")
plt.show()





