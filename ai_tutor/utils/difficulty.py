def bloom_difficulty(mastery: float, pace: float = 1.0) -> str:
    """
    Map Bayesian mastery [0–1] × pace factor → difficulty bucket.
    pace < 1  => shift toward easier; pace > 1 => harder
    """
    adjusted = min(max(mastery * pace, 0), 1)
    if adjusted < 0.3:
        return "easy"
    if adjusted < 0.6:
        return "medium"
    return "hard" 