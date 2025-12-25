import numpy as np

"""
5. Build out the backporagagtion logic [NOT STARTED]
6. Build out the training loop  [NOT STARTED]
6. Write a post detailing:
    - The math behind the RNN 
    - The code behind the RNN 
    - Patterns emergent behaviors etc.
    - Interesting findings
"""

# Load the Stack overflow dataset and create mappings
def load_data():
    data = open('data/input.txt', 'r').read() 
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    return data, chars, data_size, vocab_size

data, chars, data_size, vocab_size = load_data()
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

print("data has %d characters, %d unique." % (data_size, vocab_size))


# Create the model class
class RNN: 

    def __init__(self, hidden_size, input_size, output_size):
        # Initialize model parameters
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01 # Input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01 # Hidden to hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.01 # Hidden to output
        self.bh = np.zeros((hidden_size, 1)) # Hidden bias
        self.by = np.zeros((output_size, 1)) # Output bias
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
    
    def forward(self, input_sequence, target_sequence, previous_hidden_state): 
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(previous_hidden_state)
        loss = 0 

        for t in range(len(input_sequence)):
            xs[t] = np.zeros((self.input_size, 1))
            xs[t][input_sequence[t]] = 1 
            
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh)
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            loss += -np.log(ps[t][target_sequence[t]])
        
        return loss, ps, hs  
        
# Create 1-layer RNN backpropagation pass 


# Train model 


# Evaluate model 


# Generate text for experimentation 




