"""
Extract fetched data from pkl file.
"""
from paths import *
from pickle import load


if __name__ == '__main__':
    input_dir = DB_DIR
    input_file = 'data_dict.pkl'
    output_dir = RAW_DATA_DIR

    with open(input_dir / input_file, mode='rb') as f:
        data = load(f)
        failed = []
        for k, v in data.items():
            print(f'{len(v)} items in {k}')
            try:
                with open(output_dir / f"{k.replace('/', '-')}.txt", mode='w', encoding='utf-8') as f:
                    for line in v:
                        f.write(f'{line}\n')
            except OSError:
                failed.append(k)

        if failed:
            print('\nCould not create files:')
            for f in failed:
                print(f)
