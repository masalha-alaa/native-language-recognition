"""
See word tokens which correspond to given pos trigrams (mostly for debug)
"""


from pickle import load
from paths import *
from common import *


path_p = POS_CHUNKS_DIR
path_t = TOKEN_CHUNKS_DIR

country = 'Poland'
tri = 'NN IN DT'

pos_chunks = load(open(path_p / f'{country}{PKL_LST_EXT}', mode='rb'))
tok_chunks = load(open(path_t / f'{country}{PKL_LST_EXT}', mode='rb'))

final_res = []
for i, chunk in enumerate(pos_chunks):
    for j, sentence in enumerate(chunk):
        split_s = pos_chunks[i][j].split()
        res = [i for i in range(len(split_s)-2) if ' '.join(split_s[i:i+3]) == tri]
        for item in res:
            final_res.append((i, j, item))

final2 = []
for ans in final_res:
    final2.append((' '.join(tok_chunks[ans[0]][ans[1]].split()[ans[2]:ans[2]+3]), tok_chunks[ans[0]][ans[1]]))

for trigram in final2:
    print(trigram)
