# Split-Complex-Valued Neural Network

The files here implement a complex- and split-complex-valued neural network. In this series of numerical experiments, we sought to explore how the network topology and algebraic type of the network parameters affected expressivity. Accordingly, we've test combinations of regular, wide, and deep networks, and real-, complex-, and split-complex-valued network weights. 

The networks are based on LeNet-5. The baseline network is LeNet-5 (with ReLU activation and softmax cross entropy loss). The wide network adds approx. 1.4x as many filters/neurons at each layer (to double the number of network parameters), and the deep network doubles the number of layers. 

The important files are:

* ``SplitComplexLeNet.ipynb``: Jupyter notebook which will execute the numerical experiments
* ``utils.py``: contains functions to run the models based on network size and algebra type
* ``layers.py``: implements TensorFlow layers overloaded to accommodate different algebra types. All three number systems are commutative, so we can simply implement the layers using parameter sharing. 


## Data Loading Functions

In addition to the files to manage the network, we also include three files to load the datasets: ``cifar10.py``, ``cifar100.py``, ``svhn.py``. Each of these contain a ``load_data()`` function which will download (if necessary) the data set, and returns the training and testing data. These will hopefully make loading these datasets very straightforward for future researchers.
