import sys

sys.path.append("..")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from scipy.spatial import ConvexHull
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from data_types import Experiment


def from_db_to_pandas(query, eliminate_classes_different=False, allow_negatives=False):
    last = ""
    headers = []
    datasets = []
    series = []
    serie = None

    for experiment in query:
        filename = experiment.file_name.rsplit('/')[-1].split(".")[0]
        if last != filename:
            last = filename
            dataset_name = filename
            datasets.append(dataset_name)
            if serie is not None:
                series.append(serie)
            serie = []
        measure = experiment.method.split('.')[-1].replace("None", "").replace("-R first-last ", "").replace(" -S", ""). \
            replace(" -W", "").replace("_", " ").replace("Distance", "").replace("Dissimilarity", "").strip()
        header = measure
        if "Kappa" in header:
            continue
        if header not in headers:
            headers.append(header)
        if eliminate_classes_different and (
                experiment.number_of_classes is None or experiment.number_of_clusters != experiment.number_of_classes):
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


def order_dataset(df):
    return df.reindex(sorted(df.columns), axis=1)


def get_dataset(database, eliminate_classes_different=False, allow_negatives=False):
    engine = create_engine(database, echo=False)
    session_class = sessionmaker(bind=engine)
    session = session_class()
    query = session.query(Experiment).order_by(Experiment.file_name, Experiment.id)
    # query = for experiment in session.query(Experiment).filter(or_(Experiment.set_id==i for i in [17])).order_by(Experiment.file_name):
    # query = for experiment in session.query(Experiment).filter(Experiment.number_of_clusters == Experiment.number_of_classes).order_by(Experiment.file_name):
    return order_dataset(from_db_to_pandas(query, eliminate_classes_different, allow_negatives))


def get_averages(df):
    dft = df.apply(lambda x: x.rank(ascending=False), axis=1)
    averages = dft.mean()
    # averages = averages.sort_values()
    return averages


def compare_individually(header, df):
    pairs = [x for x in df.columns if x != header]
    for y in pairs:
        c, p = wilcoxon(df[header], df[y], alternative="greater")
        if p < 0.1:
            print(f"{header} vs {y} -> {p}")
        else:
            c1, p1 = wilcoxon(df[y], df[header], alternative="greater")
            if p1 > 0.1:
                print(f"{header} and {y} are not different, p -> {p}")


def compare_same_item(folder1, folder2, item, eliminate_classes_different):
    firsts = get_datasets(folder1, eliminate_classes_different)
    seconds = get_datasets(folder2, eliminate_classes_different)
    message = "setting to minimal score when classes and clusters do not match" if eliminate_classes_different else \
        "mantaining values when classes and clusters do not match"
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
                    f"{folder2} is significantly better than {folder1} in {dataset} for {item} when {message} -> {p1}")
            else:
                print(
                    f"{folder1} has no significant difference to {folder2} in {dataset} for {item} when {message} -> {p}, {p1}")


def get_not_significant_group(df):
    groups = []
    for column in df.columns:
        others = [x for x in df.columns if x != column]
        group = [column]
        for o in others:
            c, p = wilcoxon(df[column], df[o], alternative="greater")
            if p > 0.1:
                c1, p1 = wilcoxon(df[o], df[column], alternative="greater")
                if p1 > 0.1:
                    group.append(o)
        if len(group) > 2:
            groups.append(group)
    return groups


def get_not_significant_groups(df):
    groups = []
    for column in df.columns:
        others = [x for x in df.columns if x != column]
        for o in others:
            c, p = wilcoxon(df[column], df[o], alternative="greater")
            if p > 0.1:
                c1, p1 = wilcoxon(df[o], df[column], alternative="greater")
                if p1 > 0.1:
                    tup = (column, o)
                    if tup not in groups and tup[::-1] not in groups:
                        groups.append(tup)
    return groups


def encircle(x, y, ax=None, **kw):
    if not ax: ax = plt.gca()
    p = np.c_[x, y]
    hull = ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices, :], **kw)
    ax.add_patch(poly)


def plot_m(f_measure_x, adjusted_rand_y, f_measure_groups, adjusted_rand_groups, colors, x_limits, y_limits, label):
    fig, ax = plt.subplots()
    ax.scatter(f_measure_x, adjusted_rand_y, c=colors)
    ax.set_xlabel(f'Average F-measure {label}')
    ax.set_ylabel(f'Average Adjusted Rand Index {label}')
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    plt.title(f"Average {label} in F-Measure and Adjusted Rand", fontsize=22)
    for i, txt in enumerate(f_measure_x.index):
        ax.text(f_measure_x[i], adjusted_rand_y[i], txt, ha='center', va='bottom')

    for group in f_measure_groups:
        x = [f_measure_x[item] for item in group]
        y = [adjusted_rand_y[item] for item in group]
        encircle(x, y, ec="blue", ls="--", alpha=0.2, fill=None)

    for group in adjusted_rand_groups:
        x = [f_measure_x[item] for item in group]
        y = [adjusted_rand_y[item] for item in group]
        encircle(x, y, ec="red", ls="--", alpha=0.2, fill=None)


def plot_m2(f_measure_x, adjusted_rand_y, f_measure_groups, adjusted_rand_groups, colors, x_limits, y_limits, label):
    fig, ax = plt.subplots()
    ax.scatter(f_measure_x, adjusted_rand_y, c=colors)
    ax.set_xlabel(f'Average F-measure {label}')
    ax.set_ylabel(f'Average Adjusted Rand Index {label}')
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    plt.title(f"Average {label} in F-Measure and Adjusted Rand", fontsize=22)
    for i, txt in enumerate(f_measure_x.index):
        ax.text(f_measure_x[i], adjusted_rand_y[i], txt, ha='center', va='bottom')

    common_groups = [x for x in f_measure_groups if x in adjusted_rand_groups or x[::-1] in adjusted_rand_groups]

    for group in common_groups:
        x = [f_measure_x[item] for item in group]
        y = [adjusted_rand_y[item] for item in group]
        ax.plot(x, y, color="blue", ls="--", alpha=0.2)


def get_datasets(folder, eliminate_classes_different: bool):
    headers = ["Eskin", "Euclidean", "Gambaryan", "Goodall", "InverseOccurenceFrequency", "LearningBased E N", "Lin",
               "LinModified Kappa", "LinModified KappaMax", "Manhattan", "OccurenceFrequency"]
    df = get_dataset(f'sqlite:///{folder}/results_testing.db', eliminate_classes_different=eliminate_classes_different)
    df2 = get_dataset(f'sqlite:///{folder}/results_training.db',
                      eliminate_classes_different=eliminate_classes_different)
    rand = get_dataset(f'sqlite:///{folder}/results_rand.db', eliminate_classes_different=eliminate_classes_different)
    adjusted_rand = get_dataset(f'sqlite:///{folder}/results_adjusted_rand.db',
                                eliminate_classes_different=eliminate_classes_different, allow_negatives=True)
    common_cols = [x for x in df2.columns if x in df.columns and x in rand.columns]
    fmeasure = df.loc[:, common_cols].append(df2.loc[:, common_cols])
    return fmeasure, rand, adjusted_rand


def do_analysis(folder, eliminate_classes_different: bool):
    fmeasure, rand, adjusted_rand = get_datasets(folder, eliminate_classes_different)
    f_avg = get_averages(fmeasure)
    rand_avg = get_averages(rand)
    ad_rand_avg = get_averages(adjusted_rand)

    # print("F-Measure")
    # CDplot(fmeasure.transpose(), higher_is_better=True, alpha=0.1, output_filename='auc.png')
    # print("Rand")
    # CDplot(rand.transpose(), higher_is_better=True, alpha=0.1, output_filename='rand.png')
    # print("Adjusted Rand")
    # CDplot(adjusted_rand.transpose(), higher_is_better=True, alpha=0.1, output_filename='adjusted_rand.png')
    plt.rcParams["figure.figsize"] = (20, 10)
    fig, ax = plt.subplots()
    ax.scatter(f_avg.index, f_avg, label="f-measure", s=20 * 4 * 2, marker="*")
    ax.scatter(f_avg.index, rand_avg, label="rand", s=20 * 4 * 2, marker="+")
    ax.scatter(f_avg.index, ad_rand_avg, label="adjusted_rand", s=20 * 4 * 2, marker=".")
    ax.set_ylabel('Average rank')
    ax.set_xlabel('Measures')
    ax.legend()
    plt.title("Average rank of measures with different criteria", fontsize=22)
    fig.autofmt_xdate()
    plt.show()
    # compare_individually("LearningBased E N", fmeasure)
    f_mean = fmeasure.mean()
    adj_rand_mean = adjusted_rand.mean()
    f_groups = get_not_significant_groups(fmeasure)
    adj_rand_groups = get_not_significant_groups(adjusted_rand)
    colors = [plt.cm.tab10(i / float(len(f_mean) - 1)) for i in range(len(f_mean))]

    plot_m2(f_mean, adj_rand_mean, f_groups, adj_rand_groups, colors, (0.4, 0.55), (0.0, 0.15), "score")

    plot_m2(f_avg, ad_rand_avg, f_groups, adj_rand_groups, colors, (2, 8), (2, 8), "rank")
