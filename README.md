# Are your explanations reliable?

## Description
This project aims the reproduce results from the paper *"'Are Your Explanations Reliable?'
Investigating the Stability of LIME in Explaining Text Classifiers by Marrying
XAI and Adversarial Attack"* [(Bnurger et al., 2023)](https://arxiv.org/pdf/2305.12351.pdf).

## Local setup
Instructions for installing dependencies.

Make sure you are using Python 3.8.17, [pyenv](https://github.com/pyenv/pyenv) can be used to manage Python versions.

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

## Using Snellius

### SSH access to Snellius
Copy a public SSH key to the server so you do not need to type the password
every time you log in.
```
ssh-copy-id -i ~/.ssh/id_ed25519.pub scur1035@@snellius.surf.nl
```

Additionaly you can add these lines to `~/.ssh/config` where you fill-in `<host-name>` and `team-user-name`:
```
Host snellius
    HostName <host-name>
    User <team-user-name>
```

After this, you should be able to ssh into Snellius using the command `ssh snellius`.

### Git on Snellius
A git repo has already been set up on Snellius. Changes can be pulled and pushed
from that repository.

### Logs
Logs should be written to `./results/slurm_logs/`

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
- `CUDA_VISIBLE_DEVICES=""` this command can be used to run code on the CPU only.
