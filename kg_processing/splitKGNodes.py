import csv
import json
import re

PATH_TO_WIKIDATA_NODES = "../data/dumps/nodes_unique.csv"
PATH_TO_ENTITIES =  "../data/dumps/nodes_entities.csv"
PATH_TO_LITERALS =  "../data/dumps/nodes_literals.csv"

#divide nodes into entities and literals
def splitNodes():
    with open(PATH_TO_WIKIDATA_NODES, "r") as fp_in:
        with open(PATH_TO_ENTITIES, "w") as out_i:
            with open(PATH_TO_LITERALS, "w") as out_l:
                ires = ""
                icount = -1
                pres = ""
                pcount = -1
                lres = ""
                lcount = -1
                line = fp_in.readline()
                while line:
                    item = line
                    line = fp_in.readline()
                    item = item[:-1]
                    if item[0] == "Q": 
                        icount += 1
                        ires += item + "\n"
                    elif item[0] == "\"":
                        lcount += 1
                        lres += item + "\n"
                   

                    if icount == 1000000:
                        out_i.write(ires)
                        ires = ""
                        icount = 0              

                    if lcount == 1000000:
                        out_l.write(lres)
                        lres = ""
                        lcount = 0
                out_i.write(ires)
                out_l.write(lres)


splitNodes()
print("Successfully stored node types separately")
