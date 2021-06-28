import csv
import json

DUMP_SPECIFICATION = "wikidata_clean"
#pre-processed wikidata dump
PATH_TO_WIKIDATA_DUMP = "/GW/qa/work/data/wikidata_clean_up/dumps/cleaned_dump.csv"#"wikidata-core-for-qa/dumps/" + DUMP_SPECIFICATION + ".csv"
PATH_TO_WIKIDATA_NODES = "../data/dumps/nodes.csv"


#get each node in the dump in a separate line
def retrieveNodes():
    with open(PATH_TO_WIKIDATA_DUMP, "r") as fp_in:
        with open(PATH_TO_WIKIDATA_NODES, "w") as fp_out: 
            rows = ""
            count = -1
            prev_s = ""
            line = fp_in.readline()
            while line:
                currentLine = line
                line = fp_in.readline()
                s,p,o = currentLine.split(",", 2)
                o = o.strip()
                if not s==prev_s and not s.startswith("P"):
                    rows += str(s) + "\n"
                    prev_s = s
                rows += str(p) + "\n"
                rows += str(o) + "\n"     
                count += 1
                if count == 1000:
                    count = 0
                    fp_out.write(rows)
                    rows = ""
                        
            fp_out.write(rows)


retrieveNodes()
print("Successfully stored nodes")
