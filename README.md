# Are your explanations reliable?

## Description
This project aims the reproduce results from the paper *"'Are Your Explanations Reliable?'
Investigating the Stability of LIME in Explaining Text Classifiers by Marrying
XAI and Adversarial Attack"* [(Bnurger et al., 2023)](https://arxiv.org/pdf/2305.12351.pdf).

## Local setup
Instructions for installing dependencies.

Make sure you are using Python 3.11.4, [pyenv](https://github.com/pyenv/pyenv) can be used to manage Python versions.

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

Fix deprecated regex flag of eli5.
```
./fix.sh
```

We can see all possible options and usage using the following command:
```
./run.sh --help
```
## Recommended hyperparameters
If you want to run the code quickly for testing or reviewing use a small number of samples `--lime-sr`. One recommended experiment for example:
```
/run.sh experiment --dataset="md_gender_bias" --model="distilbert-base-uncased" --method="xaifooler" --num=5 --lime-sr=10 --max-candidates=3 --batch-size=8
```

For training use the default hyperparameters and run:
```
./run.sh train
```

## Requirements
Ideally, we want to run our code in Python 3.11.3 as Snellius officially support this.
When upgrading to 3.11.3 make sure when creating the venv to use a Python
version 3.11.3. The same requirements.txt should be used.

### Upgrading python 3.8.17 -> 3.11.3
The current version of eli5 uses a deprecated regex flag (?u) as noted in the
regex [docs](https://docs.python.org/3/library/re.html?highlight=re%20global%20flag#flags):

This can be seen as the `/venv/lib/python3.11/site-packages/eli5/lime/textutils.py` file
as can be seen in the line: `DEFAULT_TOKEN_PATTERN = r"(?u)\b\w+\b"`. In this line of code
`(?u)` can simply be removed and all code should run as expected. I wrote a script to automate
this by executing the command `./utils/fix.sh`.

## List of useful documentation
- [textattack](https://textattack.readthedocs.io/en/latest/0_get_started/basic-Intro.html)
- [textattack_example](https://textattack.readthedocs.io/en/latest/2notebook/1_Introduction_and_Transformations.html)
- `CUDA_VISIBLE_DEVICES=""` this command can be used to run code on the CPU only.



