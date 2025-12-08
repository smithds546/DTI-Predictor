import math

# Replace with your real featurizer/model later
def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def simple_featurize(text: str) -> float:
    # Very simple placeholder score based on characters
    return sum(ord(c) for c in text) % 100 / 100.0

def simple_score(drug: str, protein: str) -> float:
    d_val = simple_featurize(drug)
    p_val = simple_featurize(protein)
    return _sigmoid((d_val + p_val) - 1)