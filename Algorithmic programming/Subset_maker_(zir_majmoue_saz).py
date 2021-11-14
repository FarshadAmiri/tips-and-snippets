# This code will construct a dictionary of every possible subset of the following set.
# set = Majmoue --- subset: Zir Majmoue

Set = [50, 90, 49, 89, 48]
n = len(Set)
d = {}


for mask in range(2**n):
    l = []
    for i in range(n):
        if mask & (2**i): #  if the i'th bit in mask is 1:
            l.append(Set[i])
    d[mask+1] = l

print(d)