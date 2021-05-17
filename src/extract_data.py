"""
Extract already fetched data from pkl file.
"""
from paths import *
from pickle import load


if __name__ == '__main__':
    input_dir = DB_DIR
    input_file = f'data{PKL_DIC_EXT}'
    output_dir = RAW_DATA_DIR

    with open(input_dir / input_file, mode='rb') as f:
        data = load(f)
        failed = []
        for k, v in data.items():
            print(f'{len(v)} items in {k}')
            try:
                with open(output_dir / f"{k.replace('/', '-')}.txt", mode='a', encoding='utf-8') as f:
                    # TODO: Can this be replaced with f.writelines ?
                    for line in v:
                        f.write(f'{line}\n')
            except OSError:
                failed.append(k)

        if failed:
            print('\nCould not create files:')
            for f in failed:
                print(f)
