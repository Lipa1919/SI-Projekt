import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time

from sklearn.naive_bayes import GaussianNB


class NaiveBayes:

    def __init__(self, df):
        self.X = df.iloc[:, :-1]
        self.Y = df.iloc[:, -1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.2)
        self.classes = np.unique(self.Y)
        self.mean_variance()
        self.prior()

    def mean_variance(self):
        self.means = self.X_train.groupby(self.y_train).mean().reset_index(drop=True)
        self.variances = self.X_train.groupby(self.y_train).var().reset_index(drop=True)

    def prior(self):
        self.prior = self.y_train.value_counts() / self.y_train.shape[0]
        self.prior = self.prior.reset_index(drop=True)

    def gauss(self, row_x):
        df_repeated = pd.concat([row_x] * len(self.classes), ignore_index=True)
        p = 1 / (np.sqrt(2 * np.pi * self.variances ** 2)) * np.exp(
            - (df_repeated - self.means) ** 2 / (2 * self.variances ** 2))
        return p

    def posterior_numerator(self, test):
        p = self.gauss(test)
        p_product = p.product(axis=1)
        posterior_numerator = p_product * self.prior
        return posterior_numerator.idxmax()

    def predict(self):
        y_answers = self.classes[self.X_test.apply(lambda row: self.posterior_numerator(row.to_frame().T), axis=1)]
        return y_answers


df = pd.read_csv('iris.csv')
count = 100

start_time = time.time()
score = 0
for _ in range(count):
    p1 = NaiveBayes(df)
    answers = p1.predict()
    score = score + accuracy_score(p1.y_test, answers)
print("--- %s seconds ---" % ((time.time() - start_time)/count))
print(score/count)

start_time = time.time()
score = 0
for _ in range(count):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = GaussianNB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = score + accuracy_score(y_test, predictions)
print("--- %s seconds ---" % ((time.time() - start_time)/count))
print(score / count)
