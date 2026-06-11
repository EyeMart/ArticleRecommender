import re
import numpy
import os
from sentence_transformers import SentenceTransformer, SimilarityFunction
from SheetParser import readTipSheets   

LOW_VALUE_PHRASES = [
    r"\bcalifornia\b",
    r"\bassembly\b",
    r"\bbill\b",
    r"\bcommittee\b",
    r"\bpasses\b",
    r"\badvances\b",
    r"\bapproves\b",
    r"\bappropriations\b",
    r"\bunanimously\b",
    r"\bdiscussed by\b",
    r"\bbipartisan\b",
    r"\bsenate\b",
    r"\bsenator\b",
    r"\bsen.\b"
]
def cleanText(text: str) -> str:
    text = text.lower()

    for phrase in LOW_VALUE_PHRASES:
        text = re.sub(phrase, "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text

def jaccard(a, b):
    a = set(a)
    b = set(b)
    if not a and not b:
        return 0.0

    return len(a & b) / len(a | b)

def createEmbeddings(titles, sumAnalysis):

  if os.path.exists("title_emb.txt") and os.path.getsize("title_emb.txt") > 0:
      cleanTitlesEmb = numpy.loadtxt("title_emb.txt")
  else:
      cleanTitlesEmb = numpy.empty((0, 384))

  if os.path.exists("sum_emb.txt") and os.path.getsize("sum_emb.txt") > 0:
      sumEmb = numpy.loadtxt("sum_emb.txt")
  else:
      sumEmb = numpy.empty((0, 384))

  cleanTitlesEmb = numpy.atleast_2d(cleanTitlesEmb)
  sumEmb = numpy.atleast_2d(sumEmb)


  if len(cleanTitlesEmb) < len(titles) or len(sumEmb) < len(sumAnalysis):
      model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

      cleanTitles = [cleanText(t) for t in titles[len(cleanTitlesEmb):]]
      newTitleEmb = model.encode(cleanTitles)

      cleanTitlesEmb = numpy.vstack([cleanTitlesEmb, newTitleEmb])
      numpy.savetxt("title_emb.txt", cleanTitlesEmb)

      newSumEmb = model.encode(sumAnalysis[len(sumEmb):])

      sumEmb = numpy.vstack([sumEmb, newSumEmb])
      numpy.savetxt("sum_emb.txt", sumEmb)

  return cleanTitlesEmb, sumEmb

def getRecommendations(queryIdx):
    tipsheets = readTipSheets()
    ids = [t["tipsheet_json_id"] for t in tipsheets]
    titles = [t["title"] for t in tipsheets]
    tags = [
        [tag["tagname"] for tag in t["raw_tile_json"]["tags"]]
        for t in tipsheets
    ]
    sumAnalysis = [f"{t["summary"]}" for t in tipsheets]

    cleanTitlesEmb, sumEmb = createEmbeddings(titles, sumAnalysis)

    cleanTitleScores = cleanTitlesEmb[queryIdx] @ cleanTitlesEmb.T
    sumScores = sumEmb[queryIdx] @ sumEmb.T

    tagScores = numpy.array(
        [
            jaccard(
                tags[queryIdx],
                tags[i]
            )
            for i in range(len(tipsheets))
        ]
    )
    print("COMPARING\n" + ids[queryIdx] + " -- " + titles[queryIdx] + "\n--------------------------")

    finalScore = (
        cleanTitleScores * 0.5 +
        sumScores * 0.5
    )

    filteredCandidates = numpy.where(
        (tagScores >= 0.25) 
    )[0]

    filteredCandidates = filteredCandidates[filteredCandidates != queryIdx]


    filterdFinalScores = finalScore[filteredCandidates]

    bestScores = numpy.argsort(filterdFinalScores)[::-1][:5]

    for recommendationIdx in bestScores:
        recommendationIdx = int(recommendationIdx)
        idx = int(filteredCandidates[recommendationIdx])
        score = float(filterdFinalScores[recommendationIdx])
        # if score < 0.55:
        #     break

        print(
            round(score, 3),
            cleanTitleScores[idx],
            sumScores[idx],
            ids[idx],
            titles[idx]
        )

# getRecommendations(225)

getRecommendations(100)
