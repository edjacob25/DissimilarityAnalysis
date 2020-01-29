import argparse
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from scipy.io import arff
from scipy.stats import entropy
from Results.common import get_dataset


@dataclass_json
@dataclass
class Row:
    name: str
    number_of_attributes: int
    number_of_items: int
    missing_percentage: float
    number_of_classes: int
    dominant_class: bool  # >70
    very_dominant_class: bool  # > 90
    average_values_per_attribute: float
    mode_values_per_attribute: int
    min_values_per_attribute: int
    max_values_per_attribute: int
    std_dev_values_per_attribute: float
    average_entropy: float
    mode_entropy: int
    min_entropy: int
    max_entropy: int
    std_dev_entropy: float
    winner: str


_adjusted_rand = None


def get_winner(dataset_name: str) -> str:
    global _adjusted_rand
    if _adjusted_rand is None:
        _adjusted_rand = get_dataset(
            f"sqlite:///Results/inputed/results_adjusted_rand.db",
            eliminate_classes_different=False,
            allow_negatives=True,
        )
        # _adjusted_rand = _adjusted_rand.loc[:, ["LearningBased E N", "InverseOccurenceFrequency"]]
    try:
        winner = _adjusted_rand.loc[dataset_name, :].idxmax()
    except KeyError:
        winner = _adjusted_rand.loc[f"{dataset_name.lower()}_cleaned", :].idxmax()
    if "LearningBased" in winner:
        return "LB"
    elif "Inverse" in winner:
        return "IOF"
    else:
        return "Other"


def analyze_dataset_full(dataset: Path) -> Row:
    data, meta = arff.loadarff(dataset)
    df = pd.DataFrame(data).applymap(lambda x: x.decode("utf-8")).replace(["?"], [None])
    df = df.rename(columns={"Class": "class"}, errors="ignore")
    df_nc = df.drop("class", axis=1)
    classes = df.loc[:, "class"]
    rows = df.shape[0]
    missing = ((rows - df_nc.count().min()) / rows) * 100
    dom = classes.value_counts(normalize=True)[0]
    values_per_at = df_nc.nunique()
    ent = df_nc.apply(lambda x: entropy(x.value_counts()))

    res = Row(
        name=meta.name,
        number_of_attributes=df_nc.shape[1],
        number_of_items=rows,
        missing_percentage=missing,
        number_of_classes=classes.unique().shape[0],
        dominant_class=dom > 0.7,
        very_dominant_class=dom > 0.9,
        average_values_per_attribute=values_per_at.mean(),
        mode_values_per_attribute=values_per_at.mode()[0],
        min_values_per_attribute=values_per_at.min(),
        max_values_per_attribute=values_per_at.max(),
        std_dev_values_per_attribute=values_per_at.std(),
        average_entropy=ent.mean(),
        mode_entropy=ent.mode()[0],
        min_entropy=ent.min(),
        max_entropy=ent.max(),
        std_dev_entropy=ent.std(),
        winner=get_winner(meta.name),
    )
    return res


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
                    attribute = line.split(" ")
                    name = attribute[1].rstrip()
                    att_type = ""
                    if len(attribute) > 2:
                        att_type = attribute[2].rstrip()
                    attributes.append((name, att_type, line))
                    continue
                if line_upper.startswith("@INPUTS"):
                    inputs = [x.lstrip() for x in line.split(" ", 1)[1].split(",")]
                    attributes_used = [x.rstrip() for x in inputs]
                    continue
                if line_upper.startswith("@OUTPUT"):
                    outputs = [x.lstrip() for x in line.split(" ", 1)[1].split(",")]
                    attributes_used.extend([x.rstrip() for x in outputs])
                if line_upper.startswith("@DATA"):
                    data_processing = True
            else:
                line_data = [x.lstrip() for x in line.split(",")]
                data_size += 1
                if "?" in line or "" in line_data:
                    row_missing += 1

    categorical_atts = [x for x in attributes if x[1] == "" and x[0].split("{")[0] in attributes_used]

    return (
        relation_name,
        len(attributes_used),
        len(categorical_atts),
        data_size,
        get_percentage(row_missing, data_size),
    )


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
                    attribute = line.split(" ")
                    name = attribute[1].rstrip()
                    att_type = attribute[2].rstrip()
                    attributes.append((name, att_type, line))
                    continue
                if line_upper.startswith("@DATA"):
                    data_processing = True
            else:
                line_data = [x.lstrip() for x in line.split(",")]
                data_size += 1
                if "?" in line or "" in line_data:
                    row_missing += 1
    categorical_atts = [x for x in attributes if "{" in x[2]]
    return (
        relation_name,
        len(attributes),
        len(categorical_atts),
        data_size,
        get_percentage(row_missing, data_size),
    )


def analyze_dir(dir: Path) -> pd.DataFrame:
    cols = [
        "Name",
        "Original attributes",
        "Categorical attributes",
        "No. instances",
        "Percentage of rows with missing data",
    ]
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
        elif item.suffix == ".arff" and not "cleaned" in item.stem:
            res = analyze_dataset_arff(item)
            if res[2] > 1:
                serie = pd.Series(res, cols)
                print(serie)
                df = df.append(serie, ignore_index=True)
    return df


def full_analysis(dir: Path):
    df = pd.DataFrame()
    for item in dir.iterdir():
        if item.suffix == ".arff":
            row = analyze_dataset_full(item)
            df = df.append(row.to_dict(), ignore_index=True)
    df = df.set_index("name")
    print(df)
    return df


def main():
    parser = argparse.ArgumentParser(description="Analyzes a set of datasets")
    parser.add_argument("directories", help="One or more directories to be analyzed", nargs="+")
    parser.add_argument("--save-dir", help="Where to save the resulting latex", required=False)
    parser.add_argument("--full", help="Full mode", action="store_true")

    args = parser.parse_args()
    total_df = pd.DataFrame()
    for directory in args.directories:
        path = Path(directory)

        if path.is_dir():
            if args.full:
                df = full_analysis(path)
            else:
                df = analyze_dir(path)
            total_df = total_df.append(df)
        else:
            print("Not a directory")

    if not args.full:
        total_df = total_df.sort_values(by=["Name"])
        total_df = total_df.reset_index(drop=True)
    else:
        total_df = total_df.sort_index()
        write_arff_file(total_df, "full.arff", name="Datasets")
    print(total_df)
    if args.save_dir:
        latex = Path(args.save_dir) / "latex.tex"
        total_df.to_latex(index=False, buf=latex)


def write_arff_file(dataset: pd.DataFrame, filename="dataset.arff", name="Universities"):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(f"@RELATION {name}\n\n")
        max_len = len(max(dataset.columns, key=len))
        for header in dataset.columns:
            if dataset[header].dtype == np.float64 or dataset[header].dtype == np.int64:
                column_type = "NUMERIC"
            else:
                column_type = f"{{{','.join(dataset[header].unique())}}}"

            file.write(f"@ATTRIBUTE {header.ljust(max_len)} {column_type}\n")
        file.write("\n@DATA\n")

        for _, column in dataset.iteritems():
            if column.dtype == np.object:
                pattern = re.compile(r"^(.*)$")
                dataset[column.name] = column.str.replace(pattern, r'"\1"')

        for _, row in dataset.iterrows():
            items = [str(x) for x in row]
            items = [x if x != "nan" else "?" for x in items]
            file.write(f"{', '.join(items)}\n")


if __name__ == "__main__":
    main()
