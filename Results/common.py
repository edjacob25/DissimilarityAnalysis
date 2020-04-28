import sys
from math import sqrt
from typing import Tuple, List, Set

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.append("..")
from data_types import Experiment


def from_db_to_pandas(query, eliminate_classes_different=False, allow_negatives=False) -> pd.DataFrame:
    last = ""
    headers = []
    datasets = []
    series = []
    serie = None

    for experiment in query:
        filename = experiment.file_name.rsplit("/")[-1].split(".")[0]
        if last != filename:
            last = filename
            dataset_name = filename
            datasets.append(dataset_name)
            if serie is not None:
                series.append(serie)
            serie = []
        measure = (
            experiment.method.split(".")[-1]
            .replace("None", "")
            .replace("-R first-last ", "")
            .replace(" -S", "")
            .replace(" -W", "")
            .replace("_", " ")
            .replace("Distance", "")
            .replace("Dissimilarity", "")
            .strip()
        )
        header = measure
        if "Kappa" in header:
            continue
        if header not in headers:
            headers.append(header)
        if eliminate_classes_different and (
            experiment.number_of_classes is None or experiment.number_of_clusters != experiment.number_of_classes
        ):
            if not allow_negatives:
                serie.append(0.0)
            else:
                serie.append(-1.0)
        else:
            serie.append(experiment.f_score)

    series.append(serie)
    # print(headers)
    # print(len(headers))

    return pd.DataFrame(series, index=datasets, columns=headers)


def order_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return df.reindex(sorted(df.columns), axis=1)


def get_dataset(database: str, eliminate_classes_different=False, allow_negatives=False) -> pd.DataFrame:
    engine = create_engine(database, echo=False)
    session_class = sessionmaker(bind=engine)
    session = session_class()
    query = session.query(Experiment).order_by(Experiment.file_name, Experiment.id)
    # query = for experiment in session.query(Experiment).filter(or_(Experiment.set_id==i for i in [17])).order_by(Experiment.file_name):
    # query = for experiment in session.query(Experiment).filter(Experiment.number_of_clusters == Experiment.number_of_classes).order_by(Experiment.file_name):
    return order_dataset(from_db_to_pandas(query, eliminate_classes_different, allow_negatives))


def get_averages(df: pd.DataFrame) -> pd.Series:
    dft = df.apply(lambda x: x.rank(ascending=False), axis=1)
    averages = dft.mean()
    return averages


def compare_individually(header: str, df: pd.DataFrame):
    pairs = [x for x in df.columns if x != header]

    res = []
    for y in pairs:
        c, p = wilcoxon(df[header], df[y], alternative="greater")
        if p < 0.1:
            res.append(p)
            print(f"{header} vs {y} -> {p}")
        else:
            c1, p1 = wilcoxon(df[y], df[header], alternative="less")
            if p1 > 0.1:
                res.append(p)
                print(f"{header} and {y} are not different, p -> {p}")
            else:
                res.append(p1)
    print(f"Adding {len(res)} items")
    df1 = pd.DataFrame([res], columns=pairs)
    return df1


def compare_same_item(folder1: str, folder2: str, item: str, eliminate_classes_different: bool):
    firsts = get_datasets(folder1, eliminate_classes_different)
    seconds = get_datasets(folder2, eliminate_classes_different)
    message = (
        "setting to minimal score when classes and clusters do not match"
        if eliminate_classes_different
        else "mantaining values when classes and clusters do not match"
    )
    for i, dataset in enumerate(["f-measure", "rand", "adjusted rand"]):
        df1 = firsts[i][item]
        df2 = seconds[i][item]
        # print(f"{df1.shape} - {df2.shape}")
        common_index = df1.index & df2.index
        df1 = df1.loc[common_index]
        df2 = df2.loc[common_index]
        # print(f"{df1.shape} - {df2.shape}")
        c, p = wilcoxon(df1, df2, alternative="greater")
        # print(f"{item} - {folder1} vs {item} - {folder2} -> {p}")
        if p < 0.1:
            print(f"{folder1} is significantly better than {folder2} in {dataset} for {item} when {message} -> {p}")
        else:
            c1, p1 = wilcoxon(df2, df1, alternative="greater")
            if p1 < 0.1:
                print(
                    f"{folder2} is significantly better than {folder1} in {dataset} for {item} when {message} -> {p1}"
                )
            else:
                print(
                    f"{folder1} has no significant difference to {folder2} in {dataset} for {item} when {message} -> {p}, {p1}"
                )


def get_not_significant_group(df: pd.DataFrame) -> List[Set[str]]:
    groups = []
    for column in df.columns:
        others = [x for x in df.columns if x != column]
        group = {column}
        for o in others:
            c, p = wilcoxon(df[column], df[o], alternative="greater")
            if p > 0.05:
                c1, p1 = wilcoxon(df[o], df[column], alternative="greater")
                if p1 > 0.05:
                    group.add(o)
        if len(group) > 1 and group not in groups:
            groups.append(group)
    return groups


def find_centroid(
    group: Set, x_values: pd.Series, y_values: pd.Series, y_displacement: float = 0
) -> Tuple[float, float]:
    x_avg = x_values.loc[group].mean()
    y_avg = y_values.loc[group].mean()
    return x_avg, y_avg + y_displacement


def calculate_dimension(group: Set, values: pd.Series, margin_percentage: float = 0.50) -> float:
    v_group = values.loc[group]
    high = v_group.max()
    min = v_group.min()
    distance = 2 * ((high - min) / sqrt(2))
    return distance + distance * margin_percentage


def plot_m2(
    x_values: pd.Series,
    y_values: pd.Series,
    common_groups: List[Set],
    colors: List,
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
    label: str,
):
    fig, ax = plt.subplots()
    ax.scatter(x_values, y_values, c=colors)
    ax.set_xlabel(f"Average F-measure {label}", fontsize=18)
    ax.set_ylabel(f"Average Adjusted Rand Index {label}", fontsize=18)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    # plt.title(f"Average {label} in F-Measure and Adjusted Rand", fontsize=22)

    space_up = (y_limits[1] - y_limits[0]) / 75
    for i, txt in enumerate(x_values.index):
        box = dict(edgecolor=colors[i], alpha=0.9, fill=False) if txt == "Learning Based" else None
        ax.text(x_values[i], y_values[i] + space_up, txt, ha="center", va="bottom", fontsize=14, bbox=box)

    for group in common_groups:
        centroid = find_centroid(group, x_values, y_values, space_up)
        width = calculate_dimension(group, x_values, 0.7)
        height = calculate_dimension(group, y_values)
        circle = patches.Ellipse(centroid, width, height, facecolor="none", edgecolor="black", linestyle="--")
        ax.add_patch(circle)

    plt.savefig(f"{label}.svg", format="svg", transparent=True)


def get_datasets(folder, eliminate_classes_different: bool):
    headers = [
        "Eskin",
        "Euclidean",
        "Gambaryan",
        "Goodall",
        "InverseOccurenceFrequency",
        "LearningBased E N",
        "Lin",
        "LinModified Kappa",
        "LinModified KappaMax",
        "Manhattan",
        "OccurenceFrequency",
    ]
    df = get_dataset(f"sqlite:///{folder}/results_testing.db", eliminate_classes_different=eliminate_classes_different,)
    df2 = get_dataset(
        f"sqlite:///{folder}/results_training.db", eliminate_classes_different=eliminate_classes_different,
    )
    rand = get_dataset(f"sqlite:///{folder}/results_rand.db", eliminate_classes_different=eliminate_classes_different,)
    adjusted_rand = get_dataset(
        f"sqlite:///{folder}/results_adjusted_rand.db",
        eliminate_classes_different=eliminate_classes_different,
        allow_negatives=True,
    )
    common_cols = [x for x in df2.columns if x in df.columns and x in rand.columns]
    fmeasure = df.loc[:, common_cols].append(df2.loc[:, common_cols])

    correct_strings = {
        "LearningBased E N": "Learning Based",
        "InverseOccurenceFrequency": "Inverse Occurence Frequency",
        "OccurenceFrequency": "Occurence Frequency",
    }
    fmeasure = fmeasure.rename(correct_strings, axis=1)
    rand = rand.rename(correct_strings, axis=1)
    adjusted_rand = adjusted_rand.rename(correct_strings, axis=1)

    return fmeasure, rand, adjusted_rand


def get_common_groups(f_measure_groups: List[Set[str]], adjusted_rand_groups: List[Set[str]]) -> List[Set[str]]:
    common_groups = []
    for x in f_measure_groups:
        if x in adjusted_rand_groups:
            common_groups.append(x)
        else:
            for y in adjusted_rand_groups:
                if x.issubset(y) or y.issubset(x):
                    common_groups.append(x.intersection(y))
    return common_groups


def do_analysis(folder, eliminate_classes_different: bool):
    fmeasure, rand, adjusted_rand = get_datasets(folder, eliminate_classes_different)
    f_avg = get_averages(fmeasure)
    rand_avg = get_averages(rand)
    ad_rand_avg = get_averages(adjusted_rand)

    plt.rcParams["figure.figsize"] = (10, 10)
    fig, ax = plt.subplots()
    ax.scatter(f_avg.index, f_avg, label="f-measure", s=20 * 4 * 2, marker="*")
    ax.scatter(f_avg.index, rand_avg, label="rand", s=20 * 4 * 2, marker="+")
    ax.scatter(f_avg.index, ad_rand_avg, label="adjusted_rand", s=20 * 4 * 2, marker=".")
    ax.set_ylabel("Average rank")
    ax.set_xlabel("Measures")
    ax.legend()
    plt.title("Average rank of measures with different criteria", fontsize=22)
    fig.autofmt_xdate()
    plt.savefig("ranks.png")
    plt.show()

    f_mean = fmeasure.mean()
    adj_rand_mean = adjusted_rand.mean()

    f_groups = get_not_significant_group(fmeasure)
    adj_rand_groups = get_not_significant_group(adjusted_rand)
    common_groups = get_common_groups(f_groups, adj_rand_groups)
    print(common_groups)

    colors = [plt.cm.tab10(i / float(len(f_mean) - 1)) for i in range(len(f_mean))]

    plot_m2(
        f_mean, adj_rand_mean, common_groups, colors, (0.43, 0.50), (0.035, 0.10), "score",
    )

    plot_m2(f_avg, ad_rand_avg, common_groups, colors, (2, 8), (2, 8), "rank")

    # print(pd.DataFrame([f_avg, ad_rand_avg]).to_latex(float_format="{:0.2f}".format))
    # print(pd.DataFrame([f_mean, adj_rand_mean]).to_latex(float_format="{:0.5f}".format))
    # p_values = compare_individually("LearningBased E N", fmeasure)
    # p_values = p_values.append(compare_individually("LearningBased E N", adjusted_rand))
    # print(p_values.transpose().to_latex())
