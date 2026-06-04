import re
import numpy
import json
from sentence_transformers import SentenceTransformer, SimilarityFunction
from SheetParser import readTipSheet    

LOW_VALUE_PHRASES = [
    r"\bcalifornia\b",
    r"\bassembly\b",
    r"\bbill\b",
    r"\bcommittee\b",
    r"\bpasses\b",
    r"\badvances\b",
    r"\bapproves\b",
    r"\bappropriations\b",
    r"\bunanimously\b"
]

def jaccard_similarity(a, b):
    a = set(a)
    b = set(b)
    if not a and not b:
        return 0.0

    return len(a & b) / len(a | b)

tipsheets = readTipSheet()
# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", similarity_fn_name=SimilarityFunction.EUCLIDEAN)

titles = [t["title"] for t in tipsheets]
sumAnalysis = [f"{t["summary"]}: {t["analysis"]}" for t in tipsheets]
meta = [f"{t["people"]}, {t["committee"]}, {t["trigger_texts"]}" for t in tipsheets]
tags = [json.dumps(t["raw_tile_json"]) for t in tipsheets]

titles_emb = model.encode(titles)
sum_emb = model.encode(sumAnalysis)
meta_emb = model.encode(meta)

query_idx = 0
titles_scores = titles_emb[query_idx] @ titles_emb.T
sum_scores = sum_emb[query_idx] @ sum_emb.T
meta_scores = meta_emb[query_idx] @ meta_emb.T
tag_scores = numpy.array(
    [
        jaccard_similarity(
            tags[query_idx].split(","),
            tags[i].split(",")
        )
        for i in range(len(tipsheets))
    ]
)
final = (titles_scores * 0.05) + (meta_scores * 0.05) + (sum_scores * 0.7) + (tag_scores * 0.2)

print("COMPARING\n" + str(tipsheets[query_idx]["id"]) +  " -- " + tipsheets[query_idx]["raw_tile_json"]["headline_text"] + "\n--------------------------")


for i in range(100):
    if 0.95 > final[i] > 0.45:
        print(i, ": ", final[i], str(tipsheets[i]["id"]) +  " -- " + tipsheets[i]["raw_tile_json"]["headline_text"])