#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import numpy.linalg as la
from Testing_functions import logistic_gradient, logistic_regression
from AdaAGM import AdaAGM
import matplotlib.pyplot as plt


# In[8]:


def constant_gradient(function, gradient, x_0, s, iterations):
    """
    Implementation of gradient descent with constant stepsize
    --------------
    ARGUMENTS
        function [function]: real-valued function dependent on one variable
        gradient [function]: gradient of above function dependent on same variable
        x_0 [np.array]: starting point: compatible with function and gradient
        s [float]: stepsize
        iterations [int]: maximum number of iterations
    --------------
    RETURNS
        iterates [list]: sequence of iterates produced by gradient method
    """
    iterates = []
    function_values = []
    gradient_norms = []
    
    x_curr = x_0
    
    for n in range(1, iterations):
        gradient_curr = gradient(x_curr)
        x_next = x_curr - s * gradient_curr
        
        iterates.append(x_curr)
        function_values.append(function(x_curr))
        gradient_norms.append(la.norm(gradient_curr))
        
        x_curr = x_next
        
    iterates.append(x_next)
    function_values.append(function(x_next))
    gradient_norms.append(la.norm(gradient(x_next)))
    
    return iterates, function_values, gradient_norms


# In[9]:


def Nesterov_gradient(function, gradient, x_0, s, iterations):
    """
    Implementation of Nesterov accelerated gradient descent
    --------------
    ARGUMENTS
        function [function]: real-valued function dependent on one variable
        gradient [function]: gradient of above function dependent on same variable
        x_0 [np.array]: starting point: compatible with function and gradient
        s [float]: stepsize
        iterations [int]: maximum number of iterations
    --------------
    RETURNS
        iterates [list]: sequence of iterates produced by Nesterov gradient method
    """
    iterates = []
    function_values = []
    gradient_norms = []
    
    theta_curr = 1
    y_curr = x_0
    x_curr = x_0
    
    for n in range(iterations):
        theta_next = (1 + np.sqrt(1 + 4*theta_curr))/2
        gradient_curr = gradient(x_curr)
        y_next = x_curr - s*gradient_curr
        x_next = y_next + (theta_curr - 1)/theta_next * (y_next - y_curr)
        
        iterates.append(x_curr)
        function_values.append(function(x_curr))
        gradient_norms.append(la.norm(gradient_curr))
        
        theta_curr = theta_next
        y_curr = y_next
        x_curr = x_next
        
    iterates.append(x_next)
    function_values.append(function(x_next))
    gradient_norms.append(la.norm(gradient(x_next)))
    
    return iterates, function_values, gradient_norms


# Eher nicht verwenden, da schlechter als AdaNAG_G und in anderem Paper gar nicht getestet

# In[10]:


def AdaNAG(function, gradient, x_0, s_0, iterations):
    """
    Implementation of Adaptive Nesterov accelerated gradient descent
    --------------
    ARGUMENTS
        function [function]: real-valued function dependent on one variable
        gradient [function]: gradient of above function dependent on same variable
        x_0 [np.array]: starting point: compatible with function and gradient
        s_0 [float]: starting stepsize
        iterations [int]: maximum number of iterations
    --------------
    RETURNS
        iterates [list]: sequence of iterates produced by gradient method
    """
    iterates = [x_0]
    theta_0 = 1
    theta_1 = 1/2 *(1+np.sqrt(1 + 4*theta_0**2))
    theta_2 = 1/2 *(1+np.sqrt(1 + 4*theta_1**2))
    theta_3 = 1/2 *(1+np.sqrt(1 + 4*theta_2**2))
    theta_4 = 1/2 *(1+np.sqrt(1 + 4*theta_3**2))
    theta_5 = 1/2 *(1+np.sqrt(1 + 4*theta_4**2))
    alpha_1 = 1/2 *(1 - 1/theta_3)
    alpha_2 = 1/2 *(1 - 1/theta_4)
    alpha_3 = 1/2 *(1 - 1/theta_5)
    alpha_0 = 2*theta_2/(theta_2 - 1) * 1/(1/alpha_3 + 1/alpha_2**2 - 1/alpha_1)

    x_k = x_0
    z_k = x_0
    s_k = s_0
    theta_k2 = theta_2
    alpha_k = alpha_0
    
    for k in range(iterations):
        theta_k3 = 1/2 *(1+np.sqrt(1 + 4*theta_k2**2))
        alpha_k1 = 1/2 *(1 - 1/theta_k2)
        
        y_k1 = x_k - s_k * gradient(x_k)
        z_k1 = z_k - s_k * alpha_k * theta_k2 * gradient(x_k)
        x_k1 = (1 - 1/theta_k3)*y_k1 + 1/theta_k3 * z_k1
        
        
        if k == 0:
            L_1 = -1/2 * la.norm(gradient(x_k1) - gradient(x_k))**2 / (function(x_k1) - function(x_k) + np.dot(gradient(x_k1), x_k - x_k1))
            s_k = np.min([alpha_0/alpha_1 *theta_2/(theta_3*(theta_3 - 1)) * s_0, alpha_2**2*alpha_3/((alpha_3 + alpha_2**2)*alpha_1) *1/L_1])
        else:
            s_k = stepsize(function, gradient, x_k, x_k1, s_k, alpha_k, alpha_k1)
            
        x_k = x_k1
        z_k = z_k1
        alpha_k = alpha_k1
        theta_k2 = theta_k3
        
        iterates.append(y_k1)
        
    return iterates


# In[11]:


def stepsize(function, gradient, x_k, x_k1, s_k, alpha_k, alpha_k1):
    """
    """

    L_k1 = -1/2 * la.norm(gradient(x_k1) - gradient(x_k))**2 / (function(x_k1) - function(x_k) + np.inner(gradient(x_k1), x_k - x_k1))
    
    return np.min([alpha_k/alpha_k1 * s_k, alpha_k**2/(alpha_k1 + alpha_k**2) * 1/L_k1])


# In[3]:


def AdaNAG_G(function, gradient, x_0, s_0, iterations, tau, alpha, B_0):
    """
    Implementation of generalized adaptive Nesterov accelerated gradient method
    --------------
    ARGUMENTS
        function [function]: real-valued function dependent on one variable
        gradient [function]: gradient of above function dependent on same variable
        x_0 [np.array]: starting point: compatible with function and gradient
        s_0 [float]: starting stepsize
        iterations [int]: maximum number of iterations
        tau [function]: definition of sequence tau_k dependent on natural number k
        alpha [function]: definition of sequence alpha_k dependent on natural number k
        B_0 [float]: starting value for sequence B_k
    --------------
    RETURNS
        iterates [list]: sequence of iterates produced by AdaNAG_G
    """
    x_curr = x_0
    z_curr = x_0
    s_curr = s_0
    
    function_curr = function(x_curr)
    gradient_curr = gradient(x_curr)
    
    iterates = [x_curr]
    function_values = [function_curr]
    gradient_norms = [la.norm(gradient_curr)]
    
    tau_curr = tau(0)
    tau_next = tau(1)
    tau_next2 = tau(2)
    alpha_curr = alpha(0)
    alpha_next = alpha(1)
    
    A_prev = 0
    A_curr = alpha_next * tau_next * (tau_next - 1)
    B_curr = B_0
    B_next = alpha_next**2 * tau_next**2 * ((tau_next -1)**2/(alpha_curr * tau_curr**2) - 1)
    
    for k in range(iterations):
        y_next = x_curr - s_curr*gradient_curr
        z_next = z_curr - s_curr*alpha_curr*tau_curr*gradient_curr
        x_next = (1 - 1/tau_next) * y_next + 1/tau_next * z_next
        
        function_next = function(x_next)
        gradient_next = gradient(x_next)
        
        denominator = (function_next - function_curr) + np.dot(gradient_next, x_curr - x_next)
        #mathematically denominator < 0, however at small scales rounding errors can cause it to become positive
        #Then fallback on negative denominator from iteration before
        if denominator <= 0:
            L_next = -1/2 * la.norm(gradient_next - gradient_curr)**2/denominator
            
        s_curr = np.min([(A_prev + alpha_curr * tau_curr)/A_curr * s_curr, 1/(A_curr/B_curr + (B_next + alpha_next**2*tau_next**2)/A_curr) * 1/L_next])
        x_curr = x_next
        z_curr = z_next
        
        function_curr = function_next
        gradient_curr = gradient_next
        
        tau_curr = tau_next
        tau_next = tau_next2
        tau_next2 = tau(k + 3)
        alpha_curr = alpha_next
        alpha_next = alpha(k + 2)
        
        A_prev = A_curr
        A_curr = alpha_next * tau_next * (tau_next - 1)
        B_curr = B_next
        B_next = alpha_next**2 * tau_next**2 * ((tau_next -1)**2/(alpha_curr * tau_curr**2) - 1)
        
        iterates.append(x_next)
        function_values.append(function_next)
        gradient_norms.append(la.norm(gradient_next))
        
    return iterates, function_values, gradient_norms


# In[ ]:




