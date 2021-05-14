import matplotlib.pyplot as plt
from paths import *
from pathlib import Path
import os
import re


if __name__ == '__main__':
    SAVE_IMAGES = True
    SHOW_IMAGES = True

    input_dir = Path(ROOT_DIR) / "dataset 2021-05-13 11-54-00/sentences"
    output_dir = Path(ROOT_DIR) / "images"

    x,y = [],[]
    for country_file in os.listdir(input_dir):
        if country_file.endswith('.txt'):
            x.append(re.sub(r'\..*', '', country_file))
            y.append(sum(1 for line in open(input_dir / country_file, encoding='utf-8')))
    y,x = zip(*sorted(zip(y,x), reverse=True))
    fig = plt.gcf()
    plt.title('Sentences by Country')
    plt.xticks(rotation=90)
    plt.xlabel('Country')
    plt.ylabel('Sentences Count')
    plt.bar(x,y)
    fig.set_size_inches((13.5, 4), forward=False)
    if SAVE_IMAGES:
        plt.savefig(f'{output_dir}/sentences.png', bbox_inches='tight', dpi=200)
    if SHOW_IMAGES:
        plt.show()
