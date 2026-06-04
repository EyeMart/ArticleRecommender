import re
import numpy
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

def getRecommendations(queryIdx):
    tipsheets = readTipSheets()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", similarity_fn_name=SimilarityFunction.EUCLIDEAN)

    titles = [t["title"] for t in tipsheets]
    tags = [
        [tag["tagname"] for tag in t["raw_tile_json"]["tags"]]
        for t in tipsheets
    ]
    ids = [t["tipsheet_json_id"] for t in tipsheets]
    cleanTitles = [cleanText(t) for t in titles]
    sumAnalysis = [f"{t["summary"]}" for t in tipsheets]

    titlesEmb = model.encode(titles)
    cleanTitlesEmb = model.encode(cleanTitles)

    sumEmb = model.encode(sumAnalysis)

    titleScores = titlesEmb[queryIdx] @ titlesEmb.T
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
        if score < 0.55:
            break

        print(
            round(score, 3),
            titleScores[idx],
            cleanTitleScores[idx],
            sumScores[idx],
            ids[idx],
            titles[idx]
        )