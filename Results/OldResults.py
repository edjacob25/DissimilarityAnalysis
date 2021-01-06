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
def get_dataset(database, eliminate_classes_different=True, old_format=True):
    engine = create_engine(database, echo=False)
    session_class = sessionmaker(bind=engine)
    session = session_class()
    query = session.query(Experiment).filter(or_(Experiment.set_id == i for i in [1, 2])).order_by(Experiment.file_name)
    if not old_format:
        query = session.query(Experiment).order_by(Experiment.file_name, Experiment.id)

    # query = for experiment in session.query(Experiment).filter(Experiment.number_of_clusters == Experiment.number_of_classes).order_by(Experiment.file_name):
    return order_dataset(from_db_to_pandas(query, eliminate_classes_different))


def get_averages(df):
    dft = df.apply(lambda x: x.rank(ascending=False), axis=1)
    averages = dft.mean()
    return averages


def compare_individually(header, df):
    pairs = [x for x in df.columns if x != header]
    for y in pairs:
        c, p = wilcoxon(df[header], df[y], alternative="greater")
        if p < 0.1:
            print(f"{header} vs {y} -> {p}")
        else:
            print(f"{header} and {y} are not different, p -> {p}")


# %% [markdown]
# # Setting to 0 when classes and clusters do not match

# %%
df = get_dataset("sqlite:///results_old_june.db", eliminate_classes_different=True)
headers = [
    "Eskin",
    "EuclideanDistance",
    "Gambaryan",
    "Goodall",
    "InverseOccurenceFrequency",
    "LearningBasedDissimilarity E N",
    "Lin",
    "LinModified Kappa",
    "LinModified KappaMax",
    "ManhattanDistance",
    "OccurenceFrequency",
]
df = df.loc[:, headers]
df1 = get_dataset("sqlite:///results_testing_input.db", eliminate_classes_different=True, old_format=False)
df = df.append(df1)

# %%
avgs = get_averages(df)
avgs

# %%
# %matplotlib inline
CDplot(df.transpose(), higher_is_better=True, alpha=0.1, output_filename="adjusted_rand.png")

# %%
compare_individually("LearningBasedDissimilarity E N", df)

# %% [markdown]
# # Mantaining values when classes and clusters do not match

# %%
df = get_dataset("sqlite:///results_old_june.db", eliminate_classes_different=False)
headers = [
    "Eskin",
    "EuclideanDistance",
    "Gambaryan",
    "Goodall",
    "InverseOccurenceFrequency",
    "LearningBasedDissimilarity E N",
    "Lin",
    "LinModified Kappa",
    "LinModified KappaMax",
    "ManhattanDistance",
    "OccurenceFrequency",
]
df = df.loc[:, headers]
df1 = get_dataset("sqlite:///results_testing_input.db", eliminate_classes_different=False, old_format=False)
df = df.append(df1)

# %%
avgs = get_averages(df)
avgs

# %%
# %matplotlib inline
CDplot(df.transpose(), higher_is_better=True, alpha=0.1, output_filename="adjusted_rand.png")

# %%
compare_individually("LearningBasedDissimilarity E N", df)

# %%
