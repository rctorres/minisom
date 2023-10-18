<h1>MiniSom<img src='https://3.bp.blogspot.com/-_6UDGEHzIrs/WSfiyjmoeRI/AAAAAAAABHw/3UQylcCBEhUfHNhf56WSHBBmQ6g_lXQhwCLcB/s320/minisom_logo.png' align='right'></h1>

Self Organizing Maps
--------------------

MiniSom is a minimalistic and PyTorch based implementation of the Self Organizing Maps (SOM). SOM is a type of Artificial Neural Network able to convert complex, nonlinear statistical relationships between high-dimensional data items into simple geometric relationships on a low-dimensional display. Minisom is designed to allow researchers to easily build on top of it and to give students the ability to quickly grasp its details.


Installation
---------------------

Just use pip:

    pip install minisom_gpu


How to use it
---------------------

In order to use MiniSom you need your data organized as a PyTorch tensor where each row corresponds to an observation like the following:

```python
import torch

data = torch.tensor([[ 0.80,  0.55,  0.22,  0.03],
        [ 0.82,  0.50,  0.23,  0.03],
        [ 0.80,  0.54,  0.22,  0.03],
        [ 0.80,  0.53,  0.26,  0.03],
        [ 0.79,  0.56,  0.22,  0.03],
        [ 0.75,  0.60,  0.25,  0.03],
        [ 0.77,  0.59,  0.22,  0.03]])
```

 Then you can train MiniSom just as follows:

```python
from minisom_gpu.som import MiniSom    
som = MiniSom(6, 6, 4, sigma=0.3, learning_rate=0.5) # initialization of 6x6 SOM
som.train(data, 100) # trains the SOM with 100 iterations
```

You can obtain the position of the winning neuron on the map for a given sample as follows:

```
som.winner(data[0])
```

For an overview of all the features implemented in minisom you can browse the following examples: https://github.com/rctorres/minisom/tree/master/examples
