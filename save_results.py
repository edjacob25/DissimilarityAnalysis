import argparse
from pathlib import Path

from common import save_results

parser = argparse.ArgumentParser(description='Saves the result files of a experiment to a zip file')
parser.add_argument('directory', help="Directory in which the cleaned datasets are")
parser.add_argument('filename', help="Filename to save")

args = parser.parse_args()

path = Path(args.directory)
if not path.is_dir():
    print("The selected path is not a directory")
    exit(1)

save_results(path.resolve(), args.filename)
