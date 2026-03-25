# Script for Corpus Linguistics

The script (`main.py`) contains a handful of very short and simple functions, followed by an if-name-main block containing the main logic. The approach relies heavily on spaCy's dependency parsing. For example, you can easily move through the dependency tree using the `.children` attribute of a token (look at https://spacy.io/usage/linguistic-features for details). This makes it really easy to find the subject/object/etc.

The script is fast because it does not parse the entire corpus at once. Instead, it finds instances of *help*, takes a chunk of the surrounding text, and only parses that chunk.

## Steup and installation
This project uses uv for dependency management.

**1. Install uv**
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Clone the repo**
```
git clone https://github.com/quincyhopper/concordance.git
```

**3. Create environment and install dependencies**
```
uv sync
```

**4. Download spaCy Language Model**
```
uv run python -m spacy download en_core_web_lg
```

**5. Convert documentation .xlsx file to parquet**
```
uv run documentation_to_parquet.py
```

## Run
To run the main script, run the following command. Note that you will need a directory called ``OldBailey`` which contains the ``Processed files`` directory and the ``Documentation.parquet`` file.
```
uv run main.py
```
