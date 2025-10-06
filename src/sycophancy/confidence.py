import numpy as np

def tail_confidence(probs, filter=0.10):
    probs = probs[-int(filter*len(probs)):]
    mask = probs > 0
    return -np.mean(np.log(probs[mask]))

def full_confidence(probs):
    mask = probs > 0
    return -np.mean(np.log(probs[mask]))

def entropy(probs):
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    return np.mean(entropy)

def compute_confidence(logits):
    probs = np.exp(logits)
    return {
        'entropy': entropy(probs),
        'tail_confidence': tail_confidence(probs),
        'full_confidence': full_confidence(probs)
    }