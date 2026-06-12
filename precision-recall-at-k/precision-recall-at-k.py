def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    top_k = recommended[:k]
    relevant_set = set(relevant)

    hits = sum(1 for x in top_k if x in relevant_set)

    precisionK = hits / k
    recallK = hits / len(relevant)

    return [precisionK, recallK]