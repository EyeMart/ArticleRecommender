import json

def readTipSheet():
    data = open("data_clean.txt", "r")
    sheets = []

    for _ in range(100):
        info = data.readline()
        sheets.append(json.loads(info))
        
    data.close()
    return sheets 
