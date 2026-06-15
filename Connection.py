#First, we install our DDDB Python interface, PyMySQL
#https://pypi.org/project/PyMySQL/
import pymysql.cursors
import os
import json
from pathlib import Path
from dotenv import load_dotenv

'''
returns records to easily manipluate data
'''
def buildRecords(row):
    (
        tile_title,
        json_id,
        tipsheet_tile_json,
    ) = row

    tileData = json.loads(tipsheet_tile_json) if tipsheet_tile_json else {}

    tags = []

    for tag in tileData.get("cmtags", []):
        tags.append(tag)


    content = []
    for item in tileData.get("content", []):
        text = item.get("text")
        if text:
            content.append(text)

    
    analysis = tileData.get("billanalysis_text", "")

    return {
        "tipsheet_json_id": json_id,
        "title": tile_title,
        "tags" : tags,
        "summary": " ".join(content),
        "analysis": analysis,
        "raw_tile_json": tileData,
    }

# DATABASE credentials
load_dotenv()
myhost = os.getenv("DB_HOST")
myuser = os.getenv("DB_USER")
mypassword = os.getenv("DB_PASS")
mydatabase = os.getenv("DB") 

connection = pymysql.connect(
    host = myhost,
    user = myuser,
    password = mypassword,
    database = mydatabase,
)

sql2 = """
SELECT tile_title, tipsheet_json_id, tipsheet_tile_json FROM Tipsheets
LIMIT 1400;
"""

data = open("data_clean.txt", "w")

with connection:
    with connection.cursor() as cursor:
      cursor.execute(sql2)
      result = cursor.fetchall()
    for res in result:
      data.write(json.dumps(buildRecords(res)) + '\n')

data.close()