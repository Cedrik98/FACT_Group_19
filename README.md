# Are your explanations reliable?

## Description
This project aims the reproduce results from the paper *"'Are Your Explanations Reliable?'
Investigating the Stability of LIME in Explaining Text Classifiers by Marrying
XAI and Adversarial Attack"* [(Bnurger et al., 2023)](https://arxiv.org/pdf/2305.12351.pdf).

## Setup
Instructions for installing dependencies.

Make sure you are using python 3.8.17, [pyenv](https://github.com/pyenv/pyenv) can be used to manage python versions.

Make a virtual environment.
```
python -m venv venv
```

Activate it.
```
source ./venv/bin/activate
```

Install necessary packages.
```
pip install -r requirements.txt
```

We can see all possible options and usage using the following command:
```
./run.sh --help
```

### Pre-commit hooks
Run the command:
```
pre-commit install
```

## List of usefull documentation
- [textattack](https://textattack.readthedocs.io/en/latest/0_get_started/basic-Intro.html)
