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
from common import do_analysis, compare_same_item

# %%
# %matplotlib inline

# %% [markdown]
# ## Plots explanation
# The next plot show the average F-Measure and Adjusted Rand Index for each measure and then the average positions in the rankings obtained by the same measures.
#
#
# For further explaining, they are connected by lines with other measures which are not significantly different using the Wilcoxon test.
# Blue lines indicate that they are not significantly different using F-Measure and red lines mean they are not significantly different using Adjusted Rand Index
#
# In the first graph, a better measure will tend to be in the upper right side, as we try to maximize both the scores of F-Measure and Adjusted Rand Index. Whereas in the second graph, a better measure will tend to be in the lower left side, as we try to minimize the rankings given by the scores F-Measure and Adjusted Rand Index.

# %% [markdown]
# # Results with no inputing, setting to minimal score when classes and clusters do not match

# %%
# do_analysis("normal", True)

# %% [markdown]
# # Results with no inputing, mantaining values when classes and clusters do not match

# %%
do_analysis("normal", False)

# %% [markdown]
# # Inputed results, setting to minimal score when classes and clusters do not match

# %%
# do_analysis("inputed", True)

# %% [markdown]
# # Inputed results, mantaining values when classes and clusters do not match

# %%
do_analysis("inputed", False)

# %% [markdown]
# # Comparing our algorithm by Wilcoxon in Normal (No inputed) vs Inputed data

# %%
compare_same_item("normal", "inputed", "Learning Based", True)

# %%
compare_same_item("normal", "inputed", "Learning Based", False)

# %%
