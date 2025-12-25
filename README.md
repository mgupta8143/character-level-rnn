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
