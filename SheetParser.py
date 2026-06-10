import json

def readTipSheets():
    data = open("data_clean.txt", "r")
    sheets = []

    while True:
        info = data.readline()
        if info == "":
            break
        sheets.append(json.loads(info))
        
    data.close()
    return sheets 
