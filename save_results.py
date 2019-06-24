from zipfile import ZipFile, ZIP_LZMA, ZIP_DEFLATED, ZIP_BZIP2

import argparse
import os
from shutil import rmtree

parser = argparse.ArgumentParser(description='Saves the result files of a experiment to a zip file')
parser.add_argument('directory', help="Directory in which the cleaned datasets are")
parser.add_argument('filename', help="Filename to save")

args = parser.parse_args()

if not os.path.isdir(args.directory):
    print("The selected path is not a directory")
    exit(1)

root_dir = os.path.abspath(args.directory)
with ZipFile(os.path.join(root_dir, args.filename), 'w', compression=ZIP_BZIP2) as zipfile:
    for directory in os.listdir(root_dir):
        dir_full_path = os.path.join(root_dir, directory)
        if os.path.isdir(dir_full_path):
            zipfile.write(dir_full_path, arcname=directory)
            for file in os.listdir(dir_full_path):
                zipfile.write(os.path.join(dir_full_path, file), arcname=os.path.join(directory, file))
