import numpy as np


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    n_train = X.shape[0]
    n_classes = W.shape[1]

    for i in range(n_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        exp_sum = 0

        loss -= correct_class_score
        dW[:, y[i]] -= X[i].T
        for j in range(n_classes):
            exp_sum += np.exp(scores[j])
        loss += np.log(exp_sum)
        for j in range(n_classes):
            dW[:, j] += X[i].T * np.exp(scores[j]) / exp_sum

    loss /= n_train
    dW /= n_train

    loss += reg * np.sum(W * W)
    dW += reg / 2 * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    N = X.shape[0]

    scores = X @ W
    correct_class_scores = scores[range(N), y]
    exp_sum = np.sum(np.exp(scores), axis=1)
    loss = np.sum(np.log(exp_sum) - correct_class_scores) / N

    EXP = np.exp(scores) / np.expand_dims(exp_sum, axis=1)
    EXP[range(N), y] -= 1
    dW = X.T @ EXP / N

    loss += reg * np.sum(W * W)
    dW += reg / 2 * W

    return loss, dW
