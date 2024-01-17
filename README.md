# Are your explanations reliable?

## Description
This project aims the reproduce results from the paper *"'Are Your Explanations Reliable?'
Investigating the Stability of LIME in Explaining Text Classifiers by Marrying
XAI and Adversarial Attack"* [(Bnurger et al., 2023)](https://arxiv.org/pdf/2305.12351.pdf).

## Local setup
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

## Using Snellius

### SSH access to Snellius
Copy a public ssh key to the server so you do not need to type the password
everytime you login.
```
ssh-copy-id -i ~/.ssh/id_ed25519.pub scur1035@@snellius.surf.nl
```

Additionaly you can add these lines to `~/.ssh/config` where you fill-in `<host-name>` and `team-user-name`:
```
Host snellius
    HostName <host-name>
    User <team-user-name>
```

After this you should be able to ssh into sellius using the command `ssh snellius`.

### Git on Snellius
A git repo has already been setup on snellius. Changes can be pulled and pushed
from that repository.

### Installing requirements

## List of usefull documentation
- [textattack](https://textattack.readthedocs.io/en/latest/0_get_started/basic-Intro.html)
