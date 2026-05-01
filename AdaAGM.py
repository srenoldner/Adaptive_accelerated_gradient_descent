#!/usr/bin/env python
# coding: utf-8

# # Adaptive Accelerated gradient method for smooth convex optimiziation

# In[5]:


import numpy as np
import numpy.linalg as la


# In[1]:


def AdaAGM(function, gradient, x_0, y_0, gamma, t_0, m, s_0, omega, delta, beta, iterations):
    """
    Implementation of adaptive accelerated gradient method: see Zepeng Wang, Juan Peypouguet: "Adaptive accelerated gradient method for smooth convex optimization" 
    --------------
    ARGUMENTS
        function [function]: real-valued function dependent on one variable
        gradient [function]: gradient of above function dependent on same variable
        x_0 [np.array]: starting point: compatible with function and gradient
        y_0 [np.array]: starting point: compatible with function and gradient
        gamma [float]: iterate- and stepsize-parameter, gamma > 0
        t_0 [float]: starting value of momentum sequence, t_0 >= 1
        m [float]: parameter of momentum sequence, 0 < m <= 1
        s_0 [float]: starting stepsize, s_0 > 0
        omega [float]: stepsize parameter, 0 <= omega < 1
        delta [float]: stepsize parameter, 0 <= delta < 1
        beta [float]: stepsize parameter, beta > 0
        iterations [int]: maximum number of iterations
    --------------
    RETURNS
        iterates [list]: sequence of iterates produced by AdaAGM
        gradient_norms [list]: sequence of norms of the gradient at every iteration
        function_values [list]: sequence of function values at every iteration
    """
    t_curr = t_0
    y_curr = y_0
    x_curr = x_0
    step_curr = s_0
    #first guess for Lipschitz constant (only used in case of numerical errors)
    L_curr = 1/s_0
    function_curr = function(x_curr)
    gradient_curr = gradient(x_curr)
    
    iterates = [x_curr]
    function_values = [function_curr]
    gradient_norms = [la.norm(gradient_curr)]
    
    for n in range(iterations):
        t_next = (m + np.sqrt(m**2 + 4*t_curr**2))/2
        

        y_next = x_curr - step_curr * gradient_curr
        x_next = (y_next + 
            (t_curr - 1)/t_next * (y_next - y_curr) +
            (gamma - 1) * t_curr/t_next * (y_next - x_curr))
        
        gradient_next = gradient(x_next)
        function_next = function(x_next)
        
        step_curr, L_next = stepsize(function_curr, function_next, gradient_curr, gradient_next, x_curr, x_next, step_curr, t_next, m, beta, gamma, omega, delta, L_curr)
        
        t_curr = t_next
        y_curr = y_next
        x_curr = x_next
        L_curr = L_next
        
        iterates.append(x_next)
        function_values.append(function_next)
        gradient_norms.append(la.norm(gradient_next))
        
        function_curr = function_next
        gradient_curr = gradient_next
        
    return iterates, gradient_norms, function_values


# In[2]:


def stepsize(function_curr, function_next, gradient_curr, gradient_next, x_curr, x_next, step_curr, t_next, m, beta, gamma, omega, delta, L_curr):
    """
    args:
        function_curr [float]: function value at iteration k
        function_next [float]: function value at iteration k + 1
        gradient_curr [float]: gradient value at iteration k
        gradient_next [float]: gradient value at iteration k + 1
        x_curr [np.array]: Iterate x_k
        x_next [np.array]: Iterate x_k+1
        step_curr [float]: stepsize at iteration k
        t_next [float]: value of momentum sequence at iteration k+1
        m [float]: parameter of momentum sequence, 0 < m <= 1
        omega [float]: stepsize parameter, 0 <= omega < 1
        delta [float]: stepsize parameter, 0 <= delta < 1
        beta [float]: stepsize parameter, beta > 0
        L_curr [float]: approximation of Lipschitz constant at iteration k
    return
        step_next [float]: stepsize at iteration k+1
        L_next [float]: approximation of Lipschitz constant at iteration k+1
    """
    A = (t_next - m)/(t_next - 1)
    B = 2/((1 + beta) * gamma) * (1 - 1/t_next)
    C = (1 - omega)/(2/B + 1/(beta * (1 - delta) * gamma * A))
    
    denominator = np.inner(gradient_next, x_next - x_curr) - (function_next - function_curr)
    if denominator > 0:
        L_next = 1/2 * np.linalg.norm(gradient_next - gradient_curr)**2 / denominator
    #fallback in case numerical errors result in denominator <= 0
    else:
        L_next = L_curr
    step_next = np.min([A*step_curr, B*step_curr, C/L_next])
    
    return step_next, L_next


# In[ ]:




