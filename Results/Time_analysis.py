# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.0
#   kernelspec:
#     display_name: thesis
#     language: python
#     name: thesis
# ---

# %%
import sys

sys.path.append("..")
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from scipy.spatial import ConvexHull
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from data_types import Experiment


def from_db_to_pandas(query):
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

        serie.append(experiment.time_taken)

    series.append(serie)
    # print(headers)
    # print(len(headers))

    return pd.DataFrame(series, index=datasets, columns=headers)


def order_dataset(df):
    return df.reindex(sorted(df.columns), axis=1)


def get_dataset(database):
    engine = create_engine(database, echo=False)
    session_class = sessionmaker(bind=engine)
    session = session_class()
    query = session.query(Experiment).order_by(Experiment.file_name, Experiment.id)
    # query = for experiment in session.query(Experiment).filter(or_(Experiment.set_id==i for i in [17])).order_by(Experiment.file_name):
    # query = for experiment in session.query(Experiment).filter(Experiment.number_of_clusters == Experiment.number_of_classes).order_by(Experiment.file_name):
    return order_dataset(from_db_to_pandas(query))


def get_datasets(folder):
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
    df = get_dataset(f"sqlite:///{folder}/results_testing.db")
    df2 = get_dataset(f"sqlite:///{folder}/results_training.db")
    rand = get_dataset(f"sqlite:///{folder}/results_rand.db")
    adjusted_rand = get_dataset(f"sqlite:///{folder}/results_adjusted_rand.db")
    common_cols = [x for x in df2.columns if x in df.columns and x in rand.columns]
    fmeasure = df.loc[:, common_cols].append(df2.loc[:, common_cols])
    return fmeasure, rand, adjusted_rand


# %%
fmeasure, rand, adjusted_rand = get_datasets("inputed")

# %%
correct_strings = {
    "LearningBased E N": "Learning Based",
    "InverseOccurenceFrequency": "Inverse Occurence Frequency",
    "OccurenceFrequency": "Occurence Frequency",
}
fmeasure = fmeasure.rename(correct_strings, axis=1)
fmeasure

# %%
f_mean = fmeasure.mean()
f_mean

# %%
plt.rcParams["figure.figsize"] = (10, 10)
fig, ax = plt.subplots()
colors = [plt.cm.tab10(i / float(len(f_mean) - 1)) for i in range(len(f_mean))]
ax.bar(f_mean.index, f_mean, color=colors, log=True, zorder=100)


for y in range(10, 440, 10):
    plt.axhline(y=y, color="gray", alpha=0.5, linewidth=0.5)

ax.set_ylabel("Average time taken (seconds)", fontsize=18)
ax.set_xlabel("Measures", fontsize=18)
ax.set_yticks([1, 10, 100, 400])
ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax.tick_params(axis="both", which="major", labelsize=14)
fig.autofmt_xdate()

for txt in f_mean.index:
    ax.text(txt, f_mean[txt], f"{f_mean[txt]:.4f} s", ha="center", va="bottom", fontsize=13)


plt.savefig(f"Time.svg", format="svg", transparent=True)
plt.show()

# %%
# (f_mean.loc["LearningBased E N"] * 100) /
f_mean.drop("Learning Based").mean()

# %%
