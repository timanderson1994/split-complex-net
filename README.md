# Split-Complex-Valued Neural Network

The goal of this project is to evaluate the effectiveness of using neural networks with weights in two-dimensional algebraic systems. This project originally began as a project for [Stanford's CS 229 course](http://cs229.stanford.edu/) but has been expanded considerably since then. 

The files here implement complex- and split-complex-valued neural networks. In this series of numerical experiments, we sought to explore how the network topology and algebraic type of the network parameters affected expressivity. Accordingly, we have tested combinations of regular, wide, and deep networks, and real-, complex-, and split-complex-valued network weights. 

The important files are:

* ``main.py``: file to run the models from the command line
* ``layers.py``: implementation of important neural network layers generalized for two dimensional algebraic systems
* ``functions.py``: implementation of relevant functions for two dimensional algebraic systems
* ``models.py``: defines the models and builds different models based on the algebraic system
* ``utils.py``: helper functions for building the models

