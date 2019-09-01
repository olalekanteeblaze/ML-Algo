import numpy as np
class adalineGD(object):
    """"
    ADAptive Linear NEuron
    ______________________
    Parameters
    _______________________
    eta: float
        learning rate between 0.0 and 1.0
    n_iter: int
        Passes over the training  data 
    _________________________
    Attribute
    ________________________
    w_: 1d_array
        weights after training
    errors_ : list
        numbers of misclassification
    """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, Y):
        """
        Fit training data
        ________________
        Parameters
        ________________
        X: {array-like} shape=[n_samples, n_features]
            Training vectors,
            where n_samples is the number of samples
            and n_features is the number of features
        Y: array-like, shape=[n_samples]
            Target value
        
        Returns
        _______________
        self: object
        """
        self.w_ = np.zeros(1 + X.shape)
        self.cost_ = []

        for _ in range(self.n_iter):
            output = self.net_input(X)
            errors = Y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Calculates activation function"""
        return self.net_input(X)
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.01, 1, -1)
