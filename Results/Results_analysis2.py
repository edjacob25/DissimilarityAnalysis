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
from pandas import DataFrame
import pandas as pd
import numpy as np
import git
import matplotlib.pyplot as plt
import random
import os
from scipy.stats import wilcoxon
from sqlalchemy import create_engine, or_, Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from do_analysis import Experiment, ExperimentSet

# %%
engine = create_engine("sqlite:///results.db")
session_class = sessionmaker(bind=engine)
session = session_class()

# %% [markdown]
# # Load data into the dataframe

# %%
last = ""
headers = []
datasets = []
series = []
serie = None
code_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
repo = git.Repo(code_directory)

for experiment in session.query(Experiment).order_by(Experiment.file_name, Experiment.id):
    # for experiment in session.query(Experiment).filter(or_(Experiment.set_id==i for i in [1])).order_by(Experiment.file_name):
    # for experiment in session.query(Experiment).filter(Experiment.number_of_clusters == Experiment.number_of_classes).order_by(Experiment.file_name):
    if last != experiment.file_name:
        last = experiment.file_name
        dataset_name = experiment.file_name.rsplit("/")[-1].split(".")[0]
        datasets.append(dataset_name)
        if serie is not None:
            series.append(serie)
        serie = []
    message = experiment.set.description
    measure = experiment.method.split(".")[-1]
    if "LearningBased" in measure:
        measure = measure.replace("LearningBasedDissimilarity", "LearningBased")
        measure = measure.replace("-R first-last ", "")
        measure = measure.replace("-S", "strategy")
        measure = measure.replace(" -W", ", multiplier weight")
        measure = measure.replace(" -n", ", normalized")
        measure = measure.replace(" -w", ", decision weight")
        measure = measure.replace(" -o", ", multiplier strategy")
        measure = measure.replace(" -t", ", multiplier")

    header = f"{measure}"
    if header not in headers:
        headers.append(header)
    if experiment.number_of_classes is None or experiment.number_of_clusters != experiment.number_of_classes:
        serie.append(0.0)
        # print(measure)
    else:
        serie.append(experiment.f_score)


series.append(serie)

df = pd.DataFrame(series, index=datasets, columns=headers)
df

# %% [markdown]
# # Rank each value against the other options, per dataset

# %%
dft = df.apply(lambda x: x.rank(ascending=False), axis=1)
dft

# %% [markdown]
# # Get the mean rank per measure and order them from best to worst

# %%
averages = dft.mean()
averages = averages.sort_values()
averages.sort_values()

# %%
looking_for = [
    "D, weight A - Learning Based using A as",
    "D, weight K - Learning Based using K as",
    "E, weight N",
    "D, weight A - Learning Based using K as",
    "D, weight K - Learning Based using A as",
    "D, weight N",
]

resultant_headers = []
for config in looking_for:
    for average in averages.index:
        if config in average:
            resultant_headers.append(average)

# resultant_headers
df2 = df[resultant_headers]
# df2.mean().sort_values()
df2.apply(lambda x: x.rank(ascending=False), axis=1).mean().sort_values()

# %% [markdown]
# # Compare the best and second best using the Wilcoxon test

# %%
best = averages[resultant_headers].sort_values().index[1]
# second_best = averages.sort_values().index[1]
second_best = averages[resultant_headers].sort_values().index[7]
print(best)
print(second_best)

c, p = wilcoxon(df[best], df[second_best], alternative="greater")
print(f"{c} --- {p}")
c, p = wilcoxon(df[second_best], df[best], alternative="less")
print(f"{c} --- {p}")

diff = [x - y for x, y in zip(df[best], df[second_best])]
print(diff)

# %% [markdown]
# # Check the differences between the best and second best

# %%
print(f"{best} | {second_best}")
for i, (x, y) in enumerate(zip(df[best], df[second_best])):
    if x != y:
        print(f"{df.index[i]}: {x} - {y}")


# %%
def only_upper(s):
    return "".join(c for c in s if c.isupper())


originals = []
for av in averages.index:
    # print(av)
    if "Original measures" == av.split(" - ")[1]:
        name = av.split(" ")[0]
        if "OccurenceFrequency" in av:
            name = only_upper(name)
        originals.append((name, averages[av]))
originals

# %%
better_than = {}
for av in averages.index:
    measure = f"{av.split(' ')[0]}"
    for o_name, o_value in originals:
        if f"{o_name}Modified" == measure:
            if o_value > averages[av]:
                if o_name not in better_than:
                    better_than[o_name] = []
                better_than[o_name].append((av, averages[av]))
better_than

# %%
better_than = {}
to_compare = "Original measures times kappa"
for i, average in enumerate(averages):
    if averages.index[i].split(" - ")[1] == to_compare:
        measure = f"{avre.split(' ')[0]}"
        print(measure)
        for o_name, o_value in originals:
            if f"{o_name}Modified" == measure:
                if o_value > averages[av]:
                    if o_name not in better_than:
                        better_than[o_name] = []
                    better_than[o_name].append((av, averages[av]))
better_than

# %%
all_colors = list(plt.cm.colors.cnames.keys())
random.seed(1000)
c = random.choices(all_colors, k=125)
fig = plt.figure(figsize=(16, 10), dpi=80)
ax = fig.add_subplot(111)
# fig, ax = plt.subplots()
# box = df.boxplot(ax=ax)

bp = ax.boxplot(df.transpose(), autorange=True, widths=0.65, patch_artist=True)
ax.margins(y=0.05)

for label in plt.gca().get_yticklabels():
    label.set_fontsize(18)  # Size here overrides font_prop
for i, box in enumerate(bp["boxes"]):
    # change outline color
    # box.set( color='#7570b3', linewidth=2)
    # change fill color
    box.set_facecolor(c[i])
    pass
for median in bp["medians"]:
    median.set(color="black")

for i, val in enumerate(averages.values):
    plt.text(
        i + 1,
        val,
        "{0:.4f}".format(val),
        horizontalalignment="center",
        verticalalignment="bottom",
        fontdict={"fontweight": 500, "size": 18},
    )

plt.ylabel("F1-Score", fontsize=20)
plt.gca().set_xticklabels(df.index, rotation=60, horizontalalignment="right", fontdict={"fontweight": 500, "size": 18})
plt.savefig("box.png", transparent=True, bbox_inches="tight")
plt.show()

# %%
df.reindex(dft.mean().sort_values().index, axis=1)

# %%
ours = "LearningBased strategy D, weight A - Learning Based using A"
measures = ["Eskin", "Gambaryan", "Goodall", "OccurenceFrequency", "InverseOccurenceFrequency", "Lin"]
for measure in measures:
    t, p = wilcoxon(df[ours], df[measure], alternative="greater")
    print(f"Ours vs {measure}, pvalue {p}")

# %%
print(df[ours] - df[measures[1]])
print()


# %%
