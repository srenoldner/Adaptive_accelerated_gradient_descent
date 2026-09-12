# Adaptive accelerated gradient descent
This code implements the gradient descent algorithms gradient descent with constant stepsize, Nesterov accelerated gradien descent, AdaNAG_G (developed by Jaewook J. Suh, Shiqian Ma in "An Adaptive and Parameter-Free Nesterov’s Accelerated Gradient Method for Convex Optimization") and the newly developed algorithm AdaAGM (Zepeng Wang, Juan Peypouquet in "Adaptive accelerated gradient method for smooth convex optimization").

These methods are tested on the model problems logistic regression and least squares using real world and random data.

# Experiments corresponding to the thesis

## Least squares

Figure 2:
- Source of Data: Testing Least squares.ipynb
- Used Algorithms: 
    - AdaAGM.py: all AdaAGM versions
    - Algorithms.py: Constant stepsize, Nesterov, AdaNAG-G_{12}, AdaNAG-G^{1/2}
- Creation of Plots: Create Graphs.ipynb


Figure 3:
- Source of Data: Testing Least squares.ipynb
- Used Algorithms: 
    - AdaAGM.py: all AdaAGM versions
    - Algorithms.py: Constant stepsize, Nesterov, AdaNAG-G_{12}, AdaNAG-G^{1/2}
- Creation of Plots: Create Graphs.ipynb

## Logistic regression

Figure 4:
- Source of Data: Testing Logistic regression.ipynb
- Used Algorithms: 
    - AdaAGM.py: all AdaAGM versions
    - Algorithms.py: Constant stepsize, Nesterov, AdaNAG-G_{12}, AdaNAG-G^{1/2}
- Creation of Plots: Create Graphs.ipynb