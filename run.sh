#!/bin/bash

RED='\033[0;31m'
NC='\033[0m'

function usage() {
    echo -e "Usage:
    help       \t\t\t- print this help message
    train      \t\t\t- train language models
    experiment \t\t\t- run various experiments
    evaluate   \t\t\t- run evaluation on experiment results"
}

if [[ $1 == ?(help|--help|-h) ]]; then
    usage
    exit 0
fi

if [[ $1 == !(train|experiment|evaluate) ]]; then
    echo -e "${RED}[!]${NC} Not a valid argument ${1}\n"
    usage
    exit 1
fi

python -m src.$1 ${@:2}
