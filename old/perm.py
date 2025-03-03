# %%
from itertools import permutations
import csv
ana = 'nslightlya'

perms = permutations(ana)

with open('/Users/AdamHarris/Desktop/perms.csv','w') as result_file:
    wr = csv.writer(result_file, dialect='excel')
    wr.writerows(perms)
# %%
perms = permutations(ana)
with open('/Users/AdamHarris/Desktop/perms_.txt', 'w') as f:
    for y in perms:
        line = "".join(y)
        f.write(f"{line}\n")
# %%
perms
# %%
