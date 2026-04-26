#!/usr/bin/env python
# coding: utf-8

# # Testing functions

# In[109]:


import numpy as np
import numpy.linalg as la
from scipy import special


# In[110]:


def least_squares(x, A, b):
    """
    Evaluates the objective function of the least squares problem
    --------------
    ARGUMENTS
        x [np.array]: column vector point of evaluation, shape: n features
        A [np.array]: matrix with data, shape: m samples, n features
        b [np.array]: row vector with data, shape: m samples
    --------------
    RETURNS
        least_squares [float]: Evaluation of objective function
    """
    num_rows, num_cols = np.shape(A)
    
    return 1/num_rows * la.norm(A@x - b)**2


# In[111]:


def least_squares_gradient(x, A, b):
    """
    Evaluates the gradient of objective function of the least squares problem
    --------------
    ARGUMENTS
        x [np.array]: column vector point of evaluation, shape: n features
        A [np.array]: matrix with data, shape: m samples, n features
        b [np.array]: row vector with data, shape: m samples
    --------------
    RETURNS
        least_squares_gradient [np.array]: Evaluation of gradient of objective function, shape: n features
    """
    num_rows, num_cols = np.shape(A)
    
    return 2/num_rows * A.T@(A@x - y)


# In[112]:


def logsig(x):
    """Compute the log-sigmoid function component-wise.
        --------------
    ARGUMENTS
        x [float],[np.array]: point of evaluation
    --------------
    RETURNS
        sigma [float],[np.array]: Evaluation of objective function
    """
    out = np.zeros_like(x)
    idx0 = x < -33
    out[idx0] = x[idx0]
    idx1 = (x >= -33) & (x < -18)
    out[idx1] = x[idx1] - np.exp(x[idx1])
    idx2 = (x >= -18) & (x < 37)
    out[idx2] = -np.log1p(np.exp(-x[idx2]))
    idx3 = x >= 37
    out[idx3] = -np.exp(-x[idx3])
    return out


def logistic_regression(x, A, b, regularization):
    """
    Numerically stable implementation: https://fa.bianp.net/blog/2019/evaluate_logistic/
    Evaluates the objective function of the logistic regression problem with regularization
    --------------
    ARGUMENTS
        x [np.array]: column vector point of evaluation, shape: n features
        A [np.array]: matrix with data, shape: m samples, n features
        b [np.array]: row vector with only 0, 1 entries, shape: m samples
        regularization [float]: regularization parameter
    --------------
    RETURNS
        logistic_regression [float]: Evaluation of objective function
    """
    z = np.dot(A, x)
    b = np.asarray(b)
    return np.mean((1 - b) * z - logsig(z)) + regularization/2 * la.norm(x)**2


# In[113]:


def expit_b(x, b):
    """Compute sigmoid(x) - b component-wise."""
    idx = x < 0
    out = np.zeros_like(x)
    exp_x = np.exp(x[idx])
    b_idx = b[idx]
    out[idx] = ((1 - b_idx) * exp_x - b_idx) / (1 + exp_x)
    exp_nx = np.exp(-x[~idx])
    b_nidx = b[~idx]
    out[~idx] = ((1 - b_nidx) - b_nidx * exp_nx) / (1 + exp_nx)
    return out


def logistic_gradient(x, A, b, regularization):
    """
    Numerically stable implementation: https://fa.bianp.net/blog/2019/evaluate_logistic/
    Evaluates the gradient of the objective function of the logistic regression problem with regularization parameter gamma
    --------------
    ARGUMENTS
        x [np.array]: column vector point of evaluation, shape: n features
        A [np.array]: matrix with data, shape: m samples, n features
        y [np.array]: row vector with only 0, 1 entries, shape: m samples
        regularization [float]: regularization parameter
    --------------
    RETURNS
        logistic_gradient [np.array]: Evaluation of gradient of objective function, shape: n features
    """
    z = A.dot(x)
    s = expit_b(z, b)
    return A.T.dot(s) / A.shape[0] + regularization*x


# In[ ]:




