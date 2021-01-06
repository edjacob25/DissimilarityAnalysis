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
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt

# %matplotlib inline

# %%
data, meta = arff.loadarff("full.arff")
df = pd.DataFrame(data).applymap(lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x)
df[["dominant_class", "very_dominant_class"]] = df[["dominant_class", "very_dominant_class"]] == 1.0

df

# %%
classes = ["IOF", "LB"]


for winner in classes:
    data = df[df.winner == winner]
    dc = {}
    for col, s in data.items():
        if col == "winner":
            continue

        if col in ["dominant_class", "very_dominant_class"]:
            dc[col] = {"percentage": 1 - s.value_counts(normalize=True)[0]}
        else:
            dc[col] = {"avg": s.mean(), "std": s.std()}
    print(dc)
    # ax.plot()

# %%
data

# %%
df.columns = [x.replace("_", " ").capitalize() for x in df.columns]

# %%
fig, ax = plt.subplots(3, 5, figsize=(20, 10))

cols = [x for x in df.columns.values if x not in ["Dominant class", "Very dominant class", "Winner"]]

for i, el in enumerate(cols):
    a = df.boxplot(el, by="Winner", ax=ax.flatten()[i])

    a.get_figure().gca().set_title("")
    a.get_figure().suptitle("")
    a.get_figure().gca().set_xlabel("")
    a.get_figure().gca().set_label("")

fig.delaxes(ax[2, 4])  # remove empty subplot
plt.tight_layout()
# plt.title = ""

plt.savefig(f"meta.svg", format="svg", transparent=True)
plt.show()
# df.boxplot(by="winner", ax=ax)

# %%
df.boxplot(by="Winner")

# %%
