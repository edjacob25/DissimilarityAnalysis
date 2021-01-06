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

from pandas import DataFrame
import pandas as pd
import numpy as np
import scipy.stats as st
import scipy as sp

# from jmetal.lab.statistical_test.critical_distance import CDplot
import matplotlib.pyplot as plt
import random
import re
import os
import git
from scipy.stats import wilcoxon
from sqlalchemy import create_engine, or_, Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from data_types import Experiment, ExperimentSet


# %%
def from_db_to_pandas(query, eliminate_classes_different=True):
    last = ""
    headers = []
    datasets = []
    series = []
    serie = None
    code_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    repo = git.Repo(code_directory)

    for experiment in query:
        filename = experiment.file_name.rsplit("/")[-1].split(".")[0]
        if last != filename:
            last = filename
            dataset_name = filename
            datasets.append(dataset_name)
            if serie is not None:
                series.append(serie)
            serie = []
        message = experiment.set.description
        measure = (
            experiment.method.split(".")[-1]
            .replace("None", "")
            .replace("-R first-last ", "")
            .replace(" -S", "")
            .replace(" -W", "")
            .replace("_", " ")
            .strip()
        )
        header = measure

        if header not in headers:
            headers.append(header)
        if eliminate_classes_different and (
            experiment.number_of_classes is None or experiment.number_of_clusters != experiment.number_of_classes
        ):
            serie.append(0.0)
        else:
            serie.append(experiment.f_score)

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


def get_averages(df):
    dft = df.apply(lambda x: x.rank(ascending=False), axis=1)
    averages = dft.mean()
    averages = averages.sort_values()
    averages.sort_values()
    return averages


# %%
df = get_dataset("sqlite:///results_mixed_datasets.db")
df

# %%
avgs = get_averages(df)
avgs

# %%
# %matplotlib inline
CDplot(df.transpose(), higher_is_better=True, alpha=0.1, output_filename="adjusted_rand.png")


# %%
def compare_individually(header, df):
    pairs = [x for x in df.columns if x != header]
    for y in pairs:
        c, p = wilcoxon(df[header], df[y], alternative="greater")
        if p < 0.1:
            print(f"{header} vs {y} -> {p}")
        else:
            print(f"{header} and {y} are not different, p -> {p}")


compare_individually("LinModified KappaMax", df)

# %%
