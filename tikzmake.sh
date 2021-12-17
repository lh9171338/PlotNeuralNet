#!/bin/bash

python $1.py
pdflatex $1.tex

rm *.aux *.log
#rm *.tex
