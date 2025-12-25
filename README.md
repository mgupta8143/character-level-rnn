# Character RNN

Inspired by Karpathy's [blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

Character-level RNN built on top of NumPy and trained on a dataset of Stack Overflow questions and answers.

## Setup

1. **Environment Setup**
   It's recommended to use a virtual environment.
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Data**
   The repository comes with a sample `data/input.txt` (Tiny Shakespeare) so you can run the model immediately.
   
   **Using Stack Overflow Data:**
   If you want to train on Stack Overflow data:
   1. Download the Stack Exchange dump (e.g., `Posts.xml`).
   2. Place `Posts.xml` in the `data/` directory.
   3. Run the parsing script:
      ```bash
      cd data
      python parse_stackoverflow.py
      cd ..
      ```

3. **Run**
   ```bash
   python rnn.py
   ```
