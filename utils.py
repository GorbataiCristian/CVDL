import random


def random_subset(s, percentage):
    out1 = []
    out2 = []
    for x in s:
        if random.random() < percentage:
            out1.append(x)
        else:
            out2.append(x)
    return out1, out2
