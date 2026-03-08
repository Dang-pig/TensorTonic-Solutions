def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    top_k = recommended[:k]
    inter = len(set(top_k).intersection(set(relevant)))
    return [inter / k, inter / len(relevant)]