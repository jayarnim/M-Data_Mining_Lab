import numpy as np
import pandas as pd
from tqdm import tqdm

class Model():
    def __init__(
            self, 
            n_factors, 
            learning_rate, 
            reg_w, 
            reg_v, 
            n_iterations,
            categorical_columns,
            categorical_pairs
            ):
        """
        Arguments
        n_factors               : 잠재요인 수
        learning_rate           : 학습률
        reg_w                   : 가중치 행렬 정규화 강도
        reg_v                   : 잠재요인 행렬 정규화 강도
        n_iterations            : 훈련 횟수
        categorical_columns     : 범주형 설명변수
        categorical_pairs       : 교차할 범주형 설명변수 쌍 리스트
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg_w = reg_w
        self.reg_v = reg_v
        self.n_iterations = n_iterations
        self.categorical_columns = categorical_columns
        self.categorical_pairs = categorical_pairs


    def fit(
            self, 
            X: pd.DataFrame, 
            y: pd.DataFrame
            ):
        """
        Arguments
        X               : Traning Data Set, Feature
        y               : Test Data Set, Label
        """
        X = self._categorical_interaction_generator(X)
        X = self._one_hot_encoding(X)
        self._params_initializer(X)

        history = []
        for iteration in tqdm(range(self.n_iterations)):
            np.random.shuffle(X)
            self._sgd(X, y)
            rmse = self._compute_rmse(X, y)
            history.append((iteration, rmse))
            print(f"Iteration: {iteration+1}, RMSE: {rmse}")

        return history


    def predict(
            self, 
            X: pd.DataFrame
            ):
        """
        Arguments
        X               : Test Data Set, Feature
        """
        linear_terms = self.b + np.dot(X, self.W)
        interactions = 0
        for key, weight in self.interaction_weights.items():
            interactions += weight * X[key]
        return linear_terms + interactions


    def _categorical_interaction_generator(self, X):
        self.interaction_columns = []
        for (feature1, feature2) in self.categorical_pairs:
            interaction_name = f'{feature1}_{feature2}_interaction'
            X[interaction_name] = X[feature1].astype(str) + "_" + X[feature2].astype(str)
            dummies = pd.get_dummies(X[interaction_name], prefix=interaction_name)
            X = pd.concat([X, dummies], axis=1)
            X = X.drop(columns=interaction_name)
            self.interaction_columns.extend(dummies.columns)

        return X


    def _one_hot_encoding(self, X):
        for column in self.categorical_columns:
            dummies = pd.get_dummies(X[column], prefix=column)
            X = pd.concat([X, dummies], axis=1)
            X = X.drop(columns=column)
        
        return X


    def _params_initializer(self, X):
        n_features = X.shape[1]
        self.b = 0
        self.W = np.zeros(n_features)
        self.interaction_weights = dict()
        for column in self.interaction_columns:
            self.interaction_weights[column] = np.random.normal(scale=1./self.n_factors)


    def _sgd(self, X, y):
        for index, row in X.iterrows():
            error = self.predict(row) - y.loc[index]
            self.b -= self.learning_rate * error
            self.W -= self.learning_rate * (error * row + 2 * self.reg_w * self.W)
            for key in self.interaction_weights:
                self.interaction_weights[key] -= self.learning_rate * (error * row[key] + 2 * self.reg_v * self.interaction_weights[key])


    def _compute_rmse(self, X, y):
        errors = []
        for index, row in X.iterrows():
            pred = self.predict(row)
            errors.append((pred - y.loc[index]) ** 2)
        return np.sqrt(np.mean(errors))
