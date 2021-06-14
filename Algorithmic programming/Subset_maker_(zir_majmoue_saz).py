# This code will construct a dictionary of every possible subset of the following set.
# set = Majmoue --- subset: Zir Majmoue

Set = [50, 90, 49, 89, 48]
n = len(Set)

for mask in range(2**n):
    l = []
    for i in range(n):
        if mask & (2**i):
            l.append(Set[i])
    d[mask+1] = l