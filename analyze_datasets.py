import argparse
import pandas as pd
from pathlib import Path
from typing import Tuple


def get_percentage(missing: int, total: int) -> str:
    return f"{(missing/total) * 100:.2f}%"


def analyze_dataset_dat(dataset: Path) -> Tuple[str, int, int, int, str]:
    attributes = []
    attributes_used = []
    data_size = 0
    row_missing = 0
    with open(dataset) as file:
        relation_name = file.readline().strip().split(" ")[-1].replace("\n", "")
        data_processing = False
        for line in file:
            if not data_processing:
                line_upper = line.upper()
                if line_upper.startswith("@ATTRIBUTE"):
                    attribute = line.split(' ')
                    name = attribute[1].rstrip()
                    att_type = ""
                    if len(attribute) > 2:
                        att_type = attribute[2].rstrip()
                    attributes.append((name, att_type, line))
                    continue
                if line_upper.startswith("@INPUTS"):
                    inputs = [x.lstrip() for x in line.split(" ", 1)[1].split(',')]
                    attributes_used = [x.rstrip() for x in inputs]
                    continue
                if line_upper.startswith("@OUTPUT"):
                    outputs = [x.lstrip() for x in line.split(" ", 1)[1].split(',')]
                    attributes_used.extend([x.rstrip() for x in outputs])
                if line_upper.startswith("@DATA"):
                    data_processing = True
            else:
                line_data = [x.lstrip() for x in line.split(',')]
                data_size += 1
                if "?" in line or "" in line_data:
                    row_missing += 1

    categorical_atts = [x for x in attributes if x[1] == "" and x[0].split('{')[0] in attributes_used]

    return relation_name, len(attributes_used), len(categorical_atts), data_size, get_percentage(row_missing, data_size)


def analyze_dataset_arff(dataset: Path) -> Tuple[str, int, int, int, str]:
    attributes = []
    data_size = 0
    row_missing = 0
    with open(dataset) as file:
        relation_name = file.readline().strip().split(" ")[-1].replace("\n", "")
        data_processing = False
        for line in file:
            if not data_processing:
                line_upper = line.upper()
                if line_upper.startswith("@ATTRIBUTE"):
                    attribute = line.split(' ')
                    name = attribute[1].rstrip()
                    att_type = attribute[2].rstrip()
                    attributes.append((name, att_type, line))
                    continue
                if line_upper.startswith("@DATA"):
                    data_processing = True
            else:
                line_data = [x.lstrip() for x in line.split(',')]
                data_size += 1
                if "?" in line or "" in line_data:
                    row_missing += 1
    categorical_atts = [x for x in attributes if "{" in x[2]]
    return relation_name, len(attributes), len(categorical_atts), data_size, get_percentage(row_missing, data_size)

def analyze_dir(dir: Path) -> pd.DataFrame:
    cols = ["Name", "Original attributes", "Categorical attributes", "No. instances",
            "Percentage of rows with missing data"]
    df = pd.DataFrame(columns=cols)
    for item in dir.iterdir():
        if item.is_dir():
            df2 = analyze_dir(item)
            df = df.append(df2)
        elif item.suffix == ".dat" and item.stem == dir.name:
            res = analyze_dataset_dat(item)
            if res[2] > 1:
                serie = pd.Series(res, cols)
                print(serie)
                df = df.append(serie, ignore_index=True)
        elif item.suffix == ".arff" and not "cleaned" in item.stem :
            res = analyze_dataset_arff(item)
            if res[2] > 1:
                serie = pd.Series(res, cols)
                print(serie)
                df = df.append(serie, ignore_index=True)
    return df


def main():
    parser = argparse.ArgumentParser(description='Analyzes a set of datasets')
    parser.add_argument('directories', help="One or more directories to be analyzed", nargs="+")
    parser.add_argument('--save-dir', help="Where to save the resulting latex", required=False)

    args = parser.parse_args()
    total_df = pd.DataFrame()
    for directory in args.directories:
        path = Path(directory)

        if path.is_dir():
            df = analyze_dir(path)
            total_df = total_df.append(df)
        else:
            print("Not a directory")
    total_df = total_df.sort_values(by=["Name"])
    total_df = total_df.reset_index(drop=True)
    print(total_df)
    if args.save_dir:
        latex = Path(args.save_dir) / "latex.tex"
        total_df.to_latex(index=False, buf=latex)


if __name__ == '__main__':
    main()
