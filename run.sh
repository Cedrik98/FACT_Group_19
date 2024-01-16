#!/bin/bash

RED='\033[0;31m'
NC='\033[0m'

if [[ $1 == !(train|experiment|evaluate) ]]; then
    echo -e "${RED}[!]${NC} Not a valid argument ${1}\n"
    echo -e "Usage:
    train      \t\t\t- train language models
    experiment \t\t\t- run various experiments
    evaluate   \t\t\t- run evaluation on experiment results"
    exit 1
fi

python -m src.$1 $@:2
