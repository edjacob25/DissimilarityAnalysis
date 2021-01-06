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
import re
import os
from scipy.stats import wilcoxon
from sqlalchemy import create_engine, or_, Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from do_analysis import Experiment, ExperimentSet


# %%
def from_db_to_pandas(query):
    last = ""
    headers = []
    datasets = []
    series = []
    serie = None
    code_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    repo = git.Repo(code_directory)

    for experiment in query:
        if last != experiment.file_name:
            last = experiment.file_name
            dataset_name = experiment.file_name.rsplit("/")[-1].split(".")[0]
            datasets.append(dataset_name)
            if serie is not None:
                series.append(serie)
            serie = []
        message = experiment.set.description
        measure = experiment.method.split(".")[-1]
        if "Learning Based" in message:
            continue

        if "Initial" not in experiment.set.description:
            header = measure.replace("Modified", "")
        else:
            header = measure

        if header not in headers:
            headers.append(header)
        if experiment.number_of_classes is None or experiment.number_of_clusters != experiment.number_of_classes:
            serie.append(0.0)
        else:
            serie.append(experiment.f_score)

    series.append(serie)
    print(len(headers))

    return pd.DataFrame(series, index=datasets, columns=headers)


# %%
engine = create_engine("sqlite:///results.db", echo=False)
session_class = sessionmaker(bind=engine)
session = session_class()
query = session.query(Experiment).order_by(Experiment.file_name, Experiment.id)
# query = for experiment in session.query(Experiment).filter(or_(Experiment.set_id==i for i in [17])).order_by(Experiment.file_name):
# query = for experiment in session.query(Experiment).filter(Experiment.number_of_clusters == Experiment.number_of_classes).order_by(Experiment.file_name):
df = from_db_to_pandas(query)
df


# %%
def parse_options(name: str):
    options = name.split(" ", 1)[1].split(" ")
    if options[1] == "K":
        weight = "Kappa"
    else:
        weight = "Auc"

    if options[3] == "B":
        strategy = "Base"
    elif options[3] == "D":
        strategy = "Discard"
    elif options[3] == "M":
        strategy = "Maximum"
    elif options[3] == "L":
        strategy = "Original value"

    if options[5] == "I":
        multiplier = "1 - weight"
    else:
        multiplier = "Normal"
    return (weight, strategy, multiplier)


def create_tag(options: str):
    weight, strategy, multiplier = parse_options(options)
    return f"{strategy} when {weight} is low and multiplying for {multiplier}"


all_columns = [x.replace("OccurenceFrequency", "OF") for x in df.columns]
all_columns = [x.replace("Inverse", "I") for x in all_columns]
df.columns = all_columns
measures = ["Eskin", "Gambaryan", "Goodall", "OF", "IOF", "Lin"]
a = []
cols = []
for measure in measures:
    pattern = re.compile(f"^{measure}\s")
    al = [x for x in all_columns if pattern.match(x)]
    other = [x for x in al if measure != x]
    b = []
    for o in other:
        col = create_tag(o)
        if col not in cols:

            cols.append(col)
        better = False
        c, p = wilcoxon(df[o], df[measure], alternative="greater")
        if p < 0.1:
            # print(f"{o} is better than base")
            b.append("+")
        else:
            c, p = wilcoxon(df[measure], df[o], alternative="less")
            if p < 0.1:
                # print(f"Base is lesser than {o}")
                b.append("+")
            else:
                b.append("-")
    a.append(b)

cols = [x.replace("Base", "Do nothing") for x in cols]
cols = [x.replace("Normal", "normal weight") for x in cols]
cols = [x.replace("Original value", "Original value(no multiplying)") for x in cols]

comp = pd.DataFrame(a, index=measures, columns=cols)
comp

# %%
better_than = {
    "weight": {"Kappa": 0, "Auc": 0},
    "action when weight is low": {
        "Base": 0,
        "Discard": 0,
        "Maximum": 0,
        "Original value": 0,
    },
    "multiplier": {
        "Normal": 0,
        "1 - weight": 0,
    },
}

for measure in measures:
    pattern = re.compile(f"^{measure}\s")
    al = [x for x in all_columns if pattern.match(x)]
    other = [x for x in al if measure != x]
    for o in other:
        better = False
        c, p = wilcoxon(df[o], df[measure], alternative="greater")
        if p < 0.1:
            print(f"{o} is better than base")
            better = True

        if better:
            weight, strategy, multiplier = parse_options(o)
            better_than["weight"][weight] += 1
            better_than["action when weight is low"][strategy] += 1
            better_than["multiplier"][multiplier] += 1

for variant in better_than:
    print(f"For {variant}:")
    for option in better_than[variant]:
        option_s = option.replace("Base", "Do nothing")
        option_s = option_s.replace("Normal", "Normal weight")
        option_s = option_s.replace("Original value", "Original value(no multiplying)")
        print(f"{option_s} is better than base measure {better_than[variant][option]} times")

# %%
df[measures]

# %%
