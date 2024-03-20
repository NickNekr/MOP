from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):

        base_model = self.base_model_class(**self.base_model_params)

        indices = np.random.choice(len(y), size=int(self.subsample * len(y)), replace=True)
        x_bootstrap = x[indices]
        y_bootstrap = y[indices]
        predictions_bootstrap = predictions[indices]

        base_model.fit(x_bootstrap, -self.loss_derivative(y_bootstrap, predictions_bootstrap))

        new_predictions = base_model.predict(x)

        optimal_gamma = self.find_optimal_gamma(y, predictions, new_predictions)

        self.models.append(base_model)
        self.gammas.append(optimal_gamma)


    def fit(self, x_train, y_train, x_valid, y_valid):
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])

        for _ in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)

            if self.early_stopping_rounds is not None:
                train_loss = self.loss_fn(y_train, train_predictions)
                valid_loss = self.loss_fn(y_valid, valid_predictions)

                self.history['train_loss'].append(train_loss)
                self.history['valid_loss'].append(valid_loss)

                if len(self.history['valid_loss']) > self.early_stopping_rounds:
                    if valid_loss >= np.min(self.history['valid_loss'][-self.early_stopping_rounds:]):
                        break


    def plot_progress(self, x_train, y_train, x_valid, y_valid):
        pass

    def predict_proba(self, x):
        raw_predictions = sum(gamma * model.predict(x) for gamma, model in zip(self.gammas, self.models))

        probabilities = self.sigmoid(raw_predictions)

        result = np.column_stack([1 - probabilities, probabilities])

        return result


    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        total_importances = np.zeros_like(self.models[0].feature_importances_)
        for model in self.models:
            total_importances += model.feature_importances_
        
        avg_importance = total_importances / len(self.models)

        importance = avg_importance / np.sum(total_importances)

        return importance 
