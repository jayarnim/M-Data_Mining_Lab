import numpy as np
from tqdm import tqdm

class Model():
    def __init__(
            self, 
            n_factors, 
            learning_rate, 
            reg_w, 
            reg_v, 
            n_iterations
            ):
        """
        Arguments
        n_factors               : 잠재요인 수
        learning_rate           : 학습률
        reg_w                   : 가중치 행렬 정규화 강도
        reg_v                   : 잠재요인 행렬 정규화 강도
        n_iterations            : 훈련 횟수
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg_w = reg_w
        self.reg_v = reg_v
        self.n_iterations = n_iterations


    def fit(self, X, y):
        """
        Arguments
        X                   : Training Data Set, Features
        y                   : Training Data Set, Label
        """
        self._params_initializer(X)

        training_process = []
        for iteration in tqdm(range(self.n_iterations)):
            np.random.shuffle(X)
            self._sgd(X, y)
            rmse = self._compute_rmse(X, y)
            training_process.append((iteration, rmse))
            print(f"Iteration: {iteration+1}, RMSE: {rmse}")

        return training_process


    def predict(self, X):
        """
        Arguments
        X                   : Test Data Set, Features
        """
        linear_terms = self.b + np.dot(X, self.W)
        interactions = 0.5 * np.sum(np.power(np.dot(X, self.V), 2) - np.dot(np.power(X, 2), np.power(self.V, 2)))
        output = linear_terms + interactions

        return output


    def _params_initializer(self, X):
        n_samples, n_features = X.shape

        self.b = 0
        self.W = np.zeros(n_features)
        self.V = np.random.normal(scale=1./self.n_factors, size=(n_features, self.n_factors))


    def _sgd(self, X, y):
        n_samples, n_features = X.shape

        for i in range(n_samples):
            error = self.predict(X[i]) - y[i]

            self.b -= self.learning_rate * error
            self.W -= self.learning_rate * (error * X[i] + 2 * self.reg_w * self.W)
            for factor in range(self.n_factors):
                x_V = np.dot(X[i], self.V[:, factor])
                x_x_V = np.dot(X[i]**2, self.V[:, factor])
                v_grad = error * (x_V - x_x_V)
                self.V[:, factor] -= self.learning_rate * (v_grad + 2 * self.reg_v * self.V[:, factor])


    def _compute_rmse(self, X, y):
        errors = []
        for i in range(X.shape[0]):
            pred = self.predict(X[i])
            errors.append((pred - y[i]) ** 2)
        
        output = np.sqrt(np.mean(errors))

        return output
