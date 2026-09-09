from paper4_kbvqa.retrieval.tfidf import TfidfRetriever

def test_retrieval_ranks_relevant_text():
    r=TfidfRetriever(["Paris France Eiffel Tower","Tokyo Japan"])
    assert r.search("Eiffel Tower France",1)[0][0]==0
