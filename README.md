# Character RNN

Inspired by Karpathy's [blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

Character-level RNN built on top of NumPy and trained on a dataset of Stack Overflow questions and answers.

**Requirements**:
- Python 3.8+
- NumPy >= 1.24.0

## Setup

1. **Environment Setup**
   It's recommended to use a virtual environment.
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Data**
   The repository comes with a sample `data/input.txt` (Hinduism Stack Exchange) so you can run the model immediately.
   
   **Using Stack Overflow Data:**
   If you want to train on Stack Overflow data:
   1. **Download Data**: Go to the [Stack Exchange Data Dump](https://archive.org/details/stackexchange). Look for `stackoverflow.com-Posts.7z` (warning: it's large, ~100GB). You can also use smaller dumps like `datascience.stackexchange.com.7z` for testing.
   2. **Extract**: unzip/7z extract to get `Posts.xml`.
   3. **Place**: Move `Posts.xml` into the `data/` directory of this repo.
   4. **Parse**:
      ```bash
      cd data
      python3 parse_stackoverflow.py
      # This generates input.txt
      cd ..
      ```

3. **Run**
   ```bash
   python3 rnn.py
   ```

## Usage Guide

### What to Expect
- **Loss**: Starts high (~100+) and should decrease to ~50 over time.
- **Samples**: 
    - At start: Random garbage (`jK@!s`).
    - ~1,000 iters: Structure emerges (`The i a te`).
    - ~100,000 iters: Words and sentence structure appear.
    - ~1M iters: Coherent text (depends on dataset size).

### Hyperparameters (in `rnn.py`)
- `hidden_size`: Number of neurons. Bigger = smarter but slower to train.
    - *Small*: 100
    - *Large*: 512+
- `seq_length`: How many steps to unroll. Longer = better memory of context (e.g. closing parenthesis), but harder to train.
    - *Standard*: 25
- `learning_rate`: How fast it learns.
    - *High*: 0.1 (Fast, but unstable)
    - *Low*: 0.001 (Slow, but precise)

### Outputs
- **loss_history.csv**: Tracks loss over time. Plot this to see your learning curve!
- **rnn_checkpoint.pkl**: Saves the model so you can resume training later.
