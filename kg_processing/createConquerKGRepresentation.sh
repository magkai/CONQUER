#!/bin/bash 

source ../ENV_conquer/bin/activate
#create conquer specific KG representation with paths augmented with qualifier information
python createConquerKGRepresentation.py
#retrieve KG nodes from dump
python retrieveKGNodes.py
#remove duplicates
awk -F, '!x[$1]++' ../data/dumps/nodes.csv > ../data/dumps/nodes_unique.csv
#split nodes into entities and literals
python splitKGNodes.py