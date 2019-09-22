import argparse
from pathlib import Path

from common import clean_experiments

parser = argparse.ArgumentParser(description='Cleans the result files and folders of a directory containing categorical'
                                             ' datasets')
parser.add_argument('directory', help="Directory in which the cleaned datasets are")

args = parser.parse_args()

path = Path(args.directory)
if not path.is_dir():
    print("The selected path is not a directory")
    exit(1)

clean_experiments(path)
