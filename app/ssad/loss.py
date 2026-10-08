def euclidean_dist(query_samples, prototype):
    """쿼리 임베딩과 기준점 사이의 거리를 구한다."""
    dists = (query_samples - prototype) ** 2
    dists = dists.sum(1).sqrt()
    return dists