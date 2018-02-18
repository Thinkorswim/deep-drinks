import numpy as np

# Can be used to generate names of length 'length'
# (default is from 2 to 4 words inclusive)
# given a list of words used in cocktail names
def generate_name(names, length=None):
    name_dist = {}
    for n in names:
        if len(n) > 1:
            name_dist[n] = name_dist.get(n, 0) + 1
    total = sum([x for x in name_dist.values()])
    keys, values = [], []
    for k in name_dist:
        keys.append(k)
        values.append(name_dist[k] / total)
    name = ''
    if not length:
        length = np.random.randint(low=2, high=5)
    for i in range(length):
        name += np.random.choice(keys, p=values).capitalize() + ' '
    return name.strip()
