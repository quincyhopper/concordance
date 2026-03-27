# Extracting non-finite complement clauses from the Old Bailey Corpus

This repo contains a script for extracting non-finite complement clauses that follow "help" in the Old Bailey Corpus. We rely on spaCy's dependency parsing for this task. To improve efficiency, we avoid parsing unnecessary files; we use `re` to find those files that contain "help" and then only parse a small snippet around each example. As a result, total run time is ~1 minute 30 seconds. 

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
