import numpy as np

"""
1. Choose a dataset of characters [DONE]
2. Build the logic to load the dataset [IN PROGRESS]
3. Build a test and training dataset in the correct split [IN PROGRESS]
4. Build out the forward pass logic [NOT STARTED]
5. Build out the backporagagtion logic [NOT STARTED]
"""

Wxx


class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.0


    def forward(self, x, hidden_state):
        pass 

rnn = RNN(1, 1, 1)
rnn.forward(np.array([1]))


"""
Let's get the first two steps, I can specify the computation
From the computation we can get further 




Previous Hidden State 



Wegight for Encoded = V  by V
Current Input One Hot Encoded = V BY 1 VECTOR

Output needs to be a V BY 1 Vector


"""

tanh(Wxh * x + Whh * hidden_state)