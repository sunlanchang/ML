from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)
# print(knn.score(X_test, y_test))
k_scores = []
k_range = range(1, 31)
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # for regression
    loss = -cross_val_score(knn, X, y, cv=10, scoring='mean_squared_error')
    # for classify
    # scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(loss.mean())

plt.plot(k_range, k_scores)
plt.xlabel('value of K')
plt.ylabel('accuracy')
plt.show()
