# FACT_Group_19

## Setup
Instructions for installing dependencies (I hope this works for everyone).

Make a virtual environment.
```python -m venv venv```

Activate it.
```source ./venv/bin/activate```

Install necessary packages.
```pip install -r requirements.txt```

We can see all options using the command.
```python main.py --help```

## List of usefull documentation
- [textattack](https://textattack.readthedocs.io/en/latest/0_get_started/basic-Intro.html)

## Issues and changes
### First issue
There were quite some difficulties getting packages to work correctly because no
versioning, etc. was available. Current versions in "requirements.txt" should
hopefully work. It also seems the newest versions of python do not work, I tried
several different versions.

### Second issue
Not too optimistic about code quality. Mixed use of tabs and spaces, so replaced
all tabs everywhere with spaces.

### Third issue
I had some trouble because rbo_threshold is required as an argument to ```main.py```
but is not actually parsed by their parser as can be seen in the file ```common.py```.
That is why I added it as an argument to the parser, does not seem necessary
anywhere else except for naming of files.

### Fourth issue
The next error I encountered was rather cryptic. It was something as follows:

```
OSError: It looks like the config file at '/home/jakob/.cache/huggingface/hub/models--cburger--BERT_HS/snapshots/a0a3036b810f8eabc07392a519e9a1610e
41c0ac/config.json' is not a valid JSON file.
```

It seems that this is correct, since some file is being downloaded from huggingface,
maybe containing some model params or something. When I tried looking up the default
model "cburger/BERT_HS" on huggingface, we can find
[this](https://huggingface.co/cburger/BERT_HS/blob/main/config.json). Which is clearly
a mistake and probably breaks the code. So for now I just replaced it with the example
used in the textattack documentation just to see if we can run the code.

### Fifth issue
The default dataset that is loaded is "hate_speech18". But this dataset does not seem
to exist in "common.py". Therefore I modified it to one of the existing options.

### Sixth issue
There were several lines in "main.py" that used the textattack library that made
little sense. Examples are the usage of the argument stopwords to the `AttackedText`
class. I even look through all previous releases to see if they might have used
an older version of the library but that does not seem to be the case. Could not
find any trace of the "stopword" argument in the source code. Same goes for the lines
following that line.

E.g.
```
# if args.min_length and example.num_words_non_stopwords < args.min_length:
#     continue
# if args.max_length and example.num_words > args.max_length:
#     continue
```

### Seventh issue
I need to use a small batchsize (I used 8) though, otherwize
CUDA runs out of memmory. You will probably need a GPU to run this code, so we should make sure
we quickly get access to snellius, we can also try getting acces to a workstation in the robolab.

### Eighth issue
targetList was not define around line 850 in "goal.py", for now just copied baseList.


# TODOs

## Git
- Setup simple pipeline to do some linting and typechecking for new code
- Protect the main branch
- Define the way we name branches etc
- Setup git project for keeping track of issues

## Report
- Setup overleaf report
- Start currating various sources that are relevant for our research

## Code
- Make sure everyone can run the code
- Understand what the code does
- Major refactoring is probably necessary for code quality etc.
- Can we use pretrained models from huggingface and all these different libs, or are we
  expected to implement more from scratch?

## Next steps
- What do we want to research?
