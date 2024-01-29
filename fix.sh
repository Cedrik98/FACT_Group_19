#!/bin/bash

FILE_NAME="./venv/lib/python3.11/site-packages/eli5/lime/textutils.py"

sed -i '17s/^[^\n]/#&/' ${FILE_NAME}
sed -i '/DEFAULT_TOKEN_PATTERN =/a \DEFAULT_TOKEN_PATTERN = r"\\b\\w+\\b"' ${FILE_NAME}
