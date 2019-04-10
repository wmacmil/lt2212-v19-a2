#!/bin/bash

python3 gendoc.py /scratch/reuters-topics/ gout1.csv
python3 gendoc.py -B 1500 /scratch/reuters-topics/ gout2.csv
python3 gendoc.py -T /scratch/reuters-topics/ gout3.csv
python3 gendoc.py -T -B 1500 /scratch/reuters-topics/ gout4.csv
python3 gendoc.py -S 100 /scratch/reuters-topics/ gout5.csv
python3 gendoc.py -S 1000 /scratch/reuters-topics/ gout6.csv
python3 gendoc.py -T -S 100 /scratch/reuters-topics/ gout7.csv
python3 gendoc.py -T -S 1000 /scratch/reuters-topics/ gout8.csv




# echo Hello World!
# ls
# echo You
# ls

