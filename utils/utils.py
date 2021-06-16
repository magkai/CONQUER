import json
import re


with open("../data/labels_dict.json") as labelFile:
    labels_dict = json.load(labelFile)


def getGoldAnswers(goldanswer_string):
    goldanswerList = goldanswer_string.split(";")
    goldanswers = []
    for ga in goldanswerList:
        if "/" in ga:
            ga = ga.rsplit("/", 1)[1]
        goldanswers.append(ga)   
    return goldanswers

# return if the given string is a timestamp
def is_timestamp(timestamp):
    pattern = re.compile('^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T00:00:00Z')
    if not(pattern.match(timestamp)):
        return False
    else:
        return True

def convertTimestamp( timestamp):
    yearPattern = re.compile('^[0-9][0-9][0-9][0-9]-00-00T00:00:00Z')
    monthPattern = re.compile('^[0-9][0-9][0-9][0-9]-[0-9][0-9]-00T00:00:00Z')
    dayPattern = re.compile('^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T00:00:00Z')
    timesplits = timestamp.split("-")
    year = timesplits[0]
    if yearPattern.match(timestamp):
        return year
    month = convertMonth(timesplits[1])
    if monthPattern.match(timestamp):
        return month + " " + year
    elif dayPattern.match(timestamp):
        day = timesplits[2].rsplit("T")[0]
        return day + " " + month + " " +year
   
    return timestamp


# convert the given month to a number
def convertMonth( month):
    return{
        "01": "january",
        "02": "february",
        "03": "march",
        "04": "april",
        "05": "may",
        "06": "june",
        "07": "july",
        "08": "august",
        "09": "september", 
        "10": "october",
        "11": "november",
        "12": "december"
    }[month]

def getLabel(entity):
    label = ""
    if entity.startswith("Q") or entity.startswith("P"):
            #for predicates: P10-23, split away counting
        if "-" in entity:
            e = entity.split("-") [0]
        else:
            e = entity
        if e in labels_dict.keys():
            label = labels_dict[e]
    else:
        if is_timestamp(entity):
            label = convertTimestamp(entity)
        elif entity.startswith("+"):
            label = entity.split("+")[1]
        else:
            label = entity

    return label

