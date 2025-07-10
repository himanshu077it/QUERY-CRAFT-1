# Contributing

## Setup
```bash
git clone https://github.com/himanshu077it/QUERY-CRAFT-1
cd QUERYCRAFT/

python3 -m venv venv
source venv/bin/activate

# install package in editable mode
pip install -e '.[all]' tox pre-commit

# Setup pre-commit hooks
pre-commit install

# List dev targets
tox list

# Run tests
tox -e py310
```

## Running the test on a Mac
```bash
tox -e mac
```

Run the necessary cells and verify that it works as expected in a real-world scenario.
