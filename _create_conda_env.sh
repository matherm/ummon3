#!/bin/bash
conda env create -f environment.yml #$@%  
#conda env export | grep -v "^prefix: " > environment.yml
#conda env export > environment.yml
#conda env update -f environment.yml #$@%  