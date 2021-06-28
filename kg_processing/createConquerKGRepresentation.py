import csv
import json

"""Create CONQUER KG Representation"""

DUMP_SPECIFICATION = "wikidata_clean"
#pre-processed wikidata dump
PATH_TO_WIKIDATA_DUMP = "wikidata-core-for-qa/dumps/" + DUMP_SPECIFICATION + ".csv"
qualifier_dict = dict()
#Dump containing triples without qualifiers
PATH_TO_TRIPLES_DUMP =  "../data/dumps/triple_only_dump.csv"
#Dump where paths between subject and object are augmented with all qualifier information
PATH_TO_S_O_QUALIFIERS_DUMP =  "../data/dumps/s_o_qualifiers_dump.csv"
#Dump where paths between subject and a qualifier object contain respective information from the main fact
PATH_TO_S_QUALIFIEROBJ_DUMP =  "../data/dumps/s_qualifierobj_dump.csv"
#Dump where paths between object and a qualifier object contain respective information from the main fact
PATH_TO_O_QUALIFIEROBJ_DUMP =  "../data/dumps/o_qualifierobj_dump.csv"
#Dump where paths between two qualifier objects contain respective information from the main fact
PATH_TO_QUALIFIEROBJ_QUALIFIEROBJ_DUMP  = "../data/dumps/qo_qo_dump.csv"

#store qualifier information per predicate
def storeQualifierPredicates():
    with open(PATH_TO_WIKIDATA_DUMP, "r") as fp_in:    
        line = fp_in.readline()
        while line:
            currentLine = line
            line = fp_in.readline()
            s,p,o = currentLine.split(",", 2)
            o = o.strip()
            if s.startswith("P"):
                if s in qualifier_dict.keys():
                    qualifier_dict[s].append([p, o])
                else:
                    qualifier_dict[s] = [[p,o]]
        
        with open('../data/dumps/qualifier_dict.json', 'w') as json_file:
            json.dump(qualifier_dict, json_file)


#store different kind of relations in separate dumps
def createRelationTypeDumps():
    with open('../data/dumps/qualifier_dict.json', 'r') as json_file:
        qualifier_dict = json.load(json_file)
        with open(PATH_TO_WIKIDATA_DUMP, "r") as fp_in:
            with open(PATH_TO_TRIPLES_DUMP, "w") as fp_triples:
                with open(PATH_TO_S_O_QUALIFIERS_DUMP, "w") as fp_qualifiers:
                    with open(PATH_TO_S_QUALIFIEROBJ_DUMP, "w") as fp_s:
                        with open(PATH_TO_O_QUALIFIEROBJ_DUMP, "w") as fp_o:
                            with open(PATH_TO_QUALIFIEROBJ_QUALIFIEROBJ_DUMP, "w") as fp_qo:
                                    rows_triples = ""
                                    rows_qualifiers = ""
                                    rows_s_qo = ""
                                    rows_o_qo = ""
                                    rows_qo_qo = ""
                                    triple_count = -1
                                    qualifiers_count = -1
                                    s_qo_count = -1
                                    o_qo_count = -1
                                    qo_qo_count = -1
                                    num_triples = 0
                                    num_qualifiers = 0
                                    num_s_qo = 0
                                    num_o_qo = 0
                                    num_qo_qo = 0
                                    line = fp_in.readline()
                                    while line:
                                        currentLine = line
                                        line = fp_in.readline()
                                        s,p,o = currentLine.split(",", 2)
                                        o = o.strip()
                                        if s.startswith("P"):
                                            continue
                                        #check if fact has qualifiers
                                        if p in qualifier_dict.keys():                    
                                            new_pred = "\"" + p
                                            qual_count = -1
                                            if not o.startswith("Q"):
                                                o_lit = o[1:-1]
                                            else:
                                                o_lit = o
                                            #go over each qualifier of the fact
                                            for qualifier in qualifier_dict[p]:
                                                qual_count += 1
                                                if not qualifier[1].startswith("Q"):
                                                    qual_lit = qualifier[1][1:-1]
                                                else:
                                                    qual_lit = qualifier[1]
                                                #collect qualifiers for fact
                                                new_pred += "|" + qualifier[0] + "_" + qual_lit
                                                #add path between subject and a qualifier object
                                                sub_to_qualifier_pred = "\"" + p + "_" + o_lit + "|" + qualifier[0] + "\""
                                                rows_s_qo += s + "," + sub_to_qualifier_pred  + "," + qualifier[1] + "\n"
                                                s_qo_count += 1
                                                #add path between object and a qualifier object
                                                obj_to_qualifier_pred = "\"" +  p + "_" + s + "|" + qualifier[0] + "\"" 
                                                rows_o_qo += o + "," + obj_to_qualifier_pred  + "," + qualifier[1] + "\n"
                                                o_qo_count += 1
                                                #add path between two qualifier objects
                                                #note: direction does not matter
                                                for i in range(qual_count+1, len(qualifier_dict[p])):
                                                    qualifier_to_qualifier_pred =  "\""  + qualifier[0] +  "|" + s + "_" + p + "_" + o_lit + "|" + qualifier_dict[p][i][0] + "\""
                                                    rows_qo_qo += qualifier[1] + "," + qualifier_to_qualifier_pred  + "," + qualifier_dict[p][i][1] + "\n"
                                                    qo_qo_count += 1
                           
                                            #add qualifier infos to main triple
                                            rows_qualifiers += s + "," + new_pred + "\"" + "," + o + "\n"
                                            qualifiers_count += 1

                                        #triple without qualifiers
                                        else:
                                            rows_triples += s + "," + p  + "," + o + "\n"
                                            triple_count += 1
                                        
                                        #write rows to respective dumps
                                        if triple_count >= 1000:
                                            num_triples += triple_count
                                            triple_count = 0
                                            fp_triples.write(rows_triples)
                                            rows_triples = ""
                                        if qualifiers_count >= 1000:
                                            num_qualifiers += qualifiers_count
                                            qualifiers_count = 0
                                            fp_qualifiers.write(rows_qualifiers)
                                            rows_qualifiers = ""
                                        if s_qo_count >= 1000:
                                            num_s_qo += s_qo_count
                                            s_qo_count = 0
                                            fp_s.write(rows_s_qo)
                                            rows_s_qo = ""
                                        if o_qo_count >= 1000:
                                            num_o_qo += o_qo_count
                                            o_qo_count = 0
                                            fp_o.write(rows_o_qo)
                                            rows_o_qo = ""        
                                        if qo_qo_count >= 1000:
                                            num_qo_qo += qo_qo_count
                                            qo_qo_count = 0
                                            fp_qo.write(rows_qo_qo)
                                            rows_qo_qo = ""
                                
                                    num_triples += triple_count
                                    num_qualifiers += qualifiers_count
                                    num_s_qo += s_qo_count
                                    num_o_qo += o_qo_count
                                    num_qo_qo += qo_qo_count
                                    fp_triples.write(rows_triples)
                                    fp_qualifiers.write(rows_qualifiers)
                                    fp_s.write(rows_s_qo)
                                    fp_o.write(rows_o_qo)
                                    fp_qo.write(rows_qo_qo) 
                                    print("number of triples: ", num_triples)
                                    print("number of s-o qualifiers: ", num_qualifiers)
                                    print("number of s to qualifier obj: ", num_s_qo)
                                    print("number of o to qualifier obj: ", num_o_qo)
                                    print("number of qualifier obj relations: ", num_qo_qo)



            
storeQualifierPredicates()
createRelationTypeDumps()
print("Successfully created dumps for CONQUER KG representation")


