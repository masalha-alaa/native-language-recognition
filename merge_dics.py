"""
Merge raw databases.
"""


from collections import defaultdict
from itertools import chain
from pickle import load, dump
from paths import *
from datetime import datetime


if __name__ == '__main__':
    paths = [
        ROOT_DIR / "dataset 2021-05-16 16-56-59/data.pkl_dic",
        ROOT_DIR / "dataset 2021-05-21 12-27-07/data.pkl_dic",
    ]

    output_dir = ROOT_DIR / f'dataset {datetime.now().strftime(DATE_STR_SHORT)}{PKL_DIC_EXT}'

    print('Program started')

    all_dics = defaultdict(list)

    dics = []
    for p in paths:
        print(f'Loading {p}')
        with open(p, mode='rb') as f:
            dics.append(load(f))

    print('Merging...')
    for k, v in chain(*zip(d.items() for d in dics)):
        all_dics[k].extend(v)

    print('Saving...')
    dump(all_dics, output_dir, -1)

    with open(ROOT_DIR / 'README.md', mode='w') as f:
        f.write('This is a merged database from the following databases:\n\n')
        for p in paths:
            f.write(f'{p}\n')

    print('Done')
