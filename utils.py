import random


def random_subset(s, percentage):
    random.shuffle(s)
    out1 = s[:int(percentage*len(s))]
    out2 = s[int(percentage*len(s)):]
    return out1, out2
