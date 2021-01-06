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
        message = repo.commit(experiment.set.commit).message.strip()
        message = experiment.set.description
        measure = experiment.method.split(".")[-1]
        if "LearningBased" in measure:
            strategy = measure.split(" ")[4]
            weight = measure.split(" ")[6]
            measure = f"LearningBased strategy {strategy}, weight {weight}"
        header = f"{measure} - {message} - {experiment.set.commit}"
        if header not in headers:
            headers.append(header)
        if experiment.number_of_classes is None or experiment.number_of_clusters != experiment.number_of_classes:
            serie.append(0.0)
        else:
            serie.append(experiment.f_score)

    series.append(serie)

    return pd.DataFrame(series, index=datasets, columns=headers)


# %%
engine = create_engine("sqlite:///results.db")
session_class = sessionmaker(bind=engine)
session = session_class()
query = session.query(Experiment).order_by(Experiment.file_name, Experiment.id)
# query = for experiment in session.query(Experiment).filter(or_(Experiment.set_id==i for i in [17])).order_by(Experiment.file_name):
# query = for experiment in session.query(Experiment).filter(Experiment.number_of_clusters == Experiment.number_of_classes).order_by(Experiment.file_name):
df = from_db_to_pandas(query)
df

# %%
dft = df.apply(lambda x: x.rank(ascending=False), axis=1)
dft

# %%
averages = dft.mean()
averages.sort_values()

# %%
strategies = {}
for i, average in enumerate(averages):
    commit = averages.index[i].split(" - ", 1)[-1]
    if commit in strategies:
        items, avg_position = strategies[commit]
        accumulated = avg_position * items
        accumulated = accumulated + average
        items = items + 1
        strategies[commit] = (items, accumulated / items)
    else:
        strategies[commit] = (1, average)
strategies


# %%
def only_upper(s):
    return "".join(c for c in s if c.isupper())


strategies = {}
measure = {}
for i, average in enumerate(averages):
    measure, commit = averages.index[i].split(" - ", 1)
    if "Original" in commit:
        if "OccurenceFrequency" in measure:
            measure = only_upper(measure)
        if commit in strategies:
            strategies[commit][measure] = average
        else:
            strategies[commit] = {measure: average}

# %%
for strategy in strategies:
    other_strategies = [x for x in strategies if x != strategy]
    better_results = 0
    for measure in strategies[strategy]:
        avg = strategies[strategy][measure]
        for other_strategy in other_strategies:
            if measure in strategies[other_strategy]:
                other_avg = strategies[other_strategy][measure]
                if avg < other_avg:
                    better_results = better_results + 1
    num_of_measures = len(strategies[strategy])
    print(
        f"Strategy {strategy} has {better_results} better results, for {num_of_measures} measures, which averages to {better_results/num_of_measures} better results per measure"
    )

# %%
