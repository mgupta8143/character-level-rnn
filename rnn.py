import numpy as np
import pickle
import os

"""
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
            loss += -np.log(ps[t][target_sequence[t]]) # Cross Entropy Loss
        
        return loss, ps, hs  

    def backward(self, input_sequence, target_sequence, ps, hs):
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        
        for t in reversed(range(len(input_sequence))):
            xs = np.zeros((self.input_size, 1))
            xs[input_sequence[t]] = 1
            
            # Output Error
            dy = np.copy(ps[t])
            dy[target_sequence[t]] -= 1 
            
            # Gradients for Output Weights
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            
            # Backpropagate into hidden state
            dh = np.dot(self.Why.T, dy) + dhnext 
            dhraw = (1 - hs[t] * hs[t]) * dh 
            
            # Gradients for Hidden Weights
            dbh += dhraw
            dWxh += np.dot(dhraw, xs.T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            
            # Save derivative for next step 
            dhnext = np.dot(self.Whh.T, dhraw)
            
        # Clip gradients to prevent "exploding gradients"
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
            
        return dWxh, dWhh, dWhy, dbh, dby
    
    def sample(self, h, seed_ix, n):
        """ 
        Sample a sequence of integers from the model 
        h is memory state, seed_ix is seed letter for first time step
        """
        x = np.zeros((self.input_size, 1))
        x[seed_ix] = 1
        
        ixes = []
        
        for t in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            
            # Use np.random.choice to sample from distribution
            ix = np.random.choice(range(self.output_size), p=p.ravel())
            
            x = np.zeros((self.input_size, 1))
            x[ix] = 1
            ixes.append(ix)
            
        return ixes
        
rnn = RNN(hidden_size=100, input_size=vocab_size, output_size=vocab_size)

# Hyperparameters
hidden_size = 100 
seq_length = 25 
learning_rate = 1e-1


mWxh, mWhh, mWhy = np.zeros_like(rnn.Wxh), np.zeros_like(rnn.Whh), np.zeros_like(rnn.Why)
mbh, mby = np.zeros_like(rnn.bh), np.zeros_like(rnn.by)

n, p = 0, 0
h_prev = np.zeros((rnn.hidden_size, 1))

while True:
    if p + seq_length + 1 >= data_size or n == 0: 
        h_prev = np.zeros((rnn.hidden_size, 1)) # reset RNN memory
        p = 0
        
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    loss, ps, hs = rnn.forward(inputs, targets, h_prev)
    dWxh, dWhh, dWhy, dbh, dby = rnn.backward(inputs, targets, ps, hs)
    
    for param, dparam, mem in zip([rnn.Wxh, rnn.Whh, rnn.Why, rnn.bh, rnn.by], 
                                  [dWxh, dWhh, dWhy, dbh, dby], 
                                  [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

    # 5. Logging & Sampling
    if n % 100 == 0:
        print(f'iter {n}, loss: {loss[0]}') # print progress
        
        # Log loss to file
        with open('loss_history.csv', 'a') as f:
            f.write(f'{n},{loss[0]}\n')
        
    if n % 1000 == 0:
        sample_ix = rnn.sample(h_prev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n %s \n----' % (txt, ))

        # Save model checkpoint
        with open('rnn_checkpoint.pkl', 'wb') as f:
            pickle.dump(rnn, f)
        print("(Checkpoint saved to rnn_checkpoint.pkl)")
        
    p += seq_length # move data pointer
    n += 1 # iteration counter
    h_prev = hs[len(inputs)-1] # keep hidden state for next batch