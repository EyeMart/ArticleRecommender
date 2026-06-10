#First, we install our DDDB Python interface, PyMySQL
#https://pypi.org/project/PyMySQL/
import pymysql.cursors
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import json

def buildRecords(row):
    (
        tip_id,
        tile_date,
        tile_title,
        tile_bill_num,
        tile_committee,
        state,
        json_id,
        tipsheet_tile_json,
        tipsheet_type
    ) = row

    tileData = json.loads(tipsheet_tile_json) if tipsheet_tile_json else {}

    tags = []

    for tag in tileData.get("cmtags", []):
        tags.append(tag)

    triggerTexts = tileData.get("trigger_texts", [])

    content = []
    for item in tileData.get("content", []):
        text = item.get("text")
        if text:
            content.append(text)

    personas_text = []
    for person in tileData.get("personas", []):
        name = f"{person.get('first', '')} {person.get('last', '')}".strip()
        info = person.get("info", "")
        note = person.get("note", "")
        affiliation = person.get("affiliation", "")

        personas_text.append(
            f"{name}. {info}. {note}. Affiliation: {affiliation}"
        )

    analysis = tileData.get("billanalysis_text", "")

    return {
        "id": tip_id,
        "tipsheet_json_id": json_id,
        "title": tile_title,
        "state": state,
        "bill": tile_bill_num,
        "committee": tile_committee,
        "date": str(tile_date),
        "tags" : tags,
        "tipsheet_type": tipsheet_type,
        "score": tileData.get("score"),
        "trigger_texts": " ".join(triggerTexts),
        "summary": " ".join(content),
        "people": " ".join(personas_text),
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
SELECT tip_id, tile_date, tile_title, tile_bill_num, tile_committee, state, tipsheet_json_id, tipsheet_tile_json, tipsheet_type FROM Tipsheets
LIMIT 800;
"""

data = open("data_clean.txt", "w")

with connection:
    with connection.cursor() as cursor:
      cursor.execute(sql2)
      result = cursor.fetchall()
    for res in result:
      data.write(json.dumps(buildRecords(res)) + '\n')

data.close()