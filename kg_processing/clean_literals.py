import csv
import json
import re

PATH_TO_LITERALS = "../data/dumps/nodes_literals.csv"
PATH_TO_CLEANED_LITERALS =  "../data/dumps/cleaned_nodes_literals.csv"

URI_PATTERN = re.compile('[A-z]*://[A-z.-/#]+.*')

def clean_literals():
    with open(PATH_TO_LITERALS, "r") as fp_in:
        with open(PATH_TO_CLEANED_LITERALS, "w") as fp_out:
            res = ""
            count = -1
            count_pruned = -1
            line = fp_in.readline()
            while line:
                item = line
                line = fp_in.readline()
                item = item[:-1]
                count += 1
                if len(item) < 2:
                    count_pruned += 1
                    continue
                if item == "\"":
                    count_pruned += 1
                    continue
                if not item[0] == "\"" and not item[0] == "Q" and not item[0] == "P":
                    if not item[0] == "L" and not item[0] == "m" and not re.match(URI_PATTERN, item):
                        print (item)
                    count_pruned += 1
                    continue
                if len(item) > 2:
                    if item[-2] == "\\" and item[-1] == "\"":
                        count_pruned += 1
                        continue
                
                res += item + "\n"
                if count == 1000000:
                    fp_out.write(res)
                    res = ""
                    count = 0
            fp_out.write(res)
            print(count)
            print(count_pruned)

clean_literals()
print("cleaned literals")
