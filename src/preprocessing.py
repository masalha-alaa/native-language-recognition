"""
Clean the data
"""
from pathlib import Path
from paths import ROOT_DIR
import os
import emoji
import re
import string
from datetime import datetime


def clean(line):
    def remove_emojis(text):
        return emoji.get_emoji_regexp().sub(r'', text)

    line = re.sub(r'\(?http\S+\)?', '', remove_emojis(line)).strip()
    if all([ch in string.punctuation for ch in list(line)]):
        line = ''
    return line.replace('[removed]', '').replace('[deleted]', '')


if __name__ == '__main__':
    ts = datetime.now()

    input_dir = Path(ROOT_DIR) / "dataset 2021-05-13 11-54-00/raw"
    output_dir = Path(ROOT_DIR) / "dataset 2021-05-13 11-54-00/clean"

    output_dir.mkdir(exist_ok=True)

    for i,filename in enumerate(os.listdir(input_dir)):
        if filename.endswith('.txt'):
            print(f'{i+1}. {filename}')
            with open(input_dir / filename, mode='r', encoding='utf-8') as fr:
                with open(output_dir / filename, mode='w', encoding='utf-8') as fw:
                    for line in fr.readlines():
                        clean_line = clean(line)
                        if clean_line:
                            fw.write(f'{clean_line}\n')

    print('')
    print(datetime.now() - ts)
