from numpy.random import seed
import numpy as np
class adalineSGD(object):
    """
    ADAptive LInear NEuron
    __________________
    Parameters
    __________________
    eta: float
        Learning rate between 0.0 and 1.0
    n_iter: int
        Passes over the training data
    ____________________
    Attributes
    ____________________
    w_: 1d array
        weights after fitting
    errors_: list
        Number of misclassifications
    shuffle: bool
        shuffles training data after every epoch
        if True to prevent cycles
    random_state: int(default: none)
        set random_state for shuffling 
        and initializing weights
    """
    def __init__(self, eta=0.01, n_iter=10,
                shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        if random_state:
            seed(random_state)
    
    def fit(self, X, y):
        """
        Fit training data
        _________________
        Parameters
        _________________
        X: {array-like}, shape = [n_samples, n_features]
            Training vectors
            where n_features is the number of n_features
            and
            n_samples is the number of samples
        Y: array-like, shape = [n_samples]
            Target value
        ______________________
        Returns
        ________________________
        self: object
        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Fit training data without initiaizing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi,target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self
        
    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to zero"""
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True
    
    def _update_weights(self, xi, target):
        """Apply Adaline rule to update the weights"""
        output = self.net_input(xi)
        error = (target - output)  
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)
