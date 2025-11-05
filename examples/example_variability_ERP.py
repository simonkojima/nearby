"""
Example: Variability of ERP in a auditory task
============================================
"""

# %%
import mne
import pandas as pd
import nearby
from moabb.datasets import Kojima2024B

# %%
"""
Extract Epochs
"""

subject = 1
l_freq = 1
h_freq = 40
resample = 128
tmin = -0.1
tmax = 1.2
baseline = [-0.05, 0.0]

dataset = Kojima2024B(task="2stream")

data = dataset.get_data(subjects=[subject])

raws = list(data[subject]["0"].values())

for raw in raws:
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="iir",
        iir_params={"ftype": "butter", "btype": "bandpass", "order": 4},
        phase="zero",
    )

raw = mne.concatenate_raws(raws)

raw.pick(picks="eeg")

epochs = mne.Epochs(raw, baseline=None, tmin=tmin, tmax=tmax)

epochs.load_data()

epochs.resample(resample)

epochs = epochs.apply_baseline(baseline=baseline)

# %%
"""
Compute Variability Metrics

# ME
"""
tmin, tmax = 0.2, 0.4

me = nearby.metrics.me(
    epochs["Target"], tmin=tmin, tmax=tmax, picks=["Cz", "CPz", "Pz"]
)

me = me["me"].mean()

# %%
"""
# SMEAT
"""
smeat = nearby.metrics.smeat(
    epochs["Target"],
    tmin=tmin,
    tmax=tmax,
    picks=["Cz", "CPz", "Pz"],
)

smeat = smeat["smeat"].mean()

# %%
"""
# WTTV
"""
wttv = nearby.metrics.wttv(
    epochs["Target"],
    tmin=tmin,
    tmax=tmax,
    picks=["Cz", "CPz", "Pz"],
)

wttv = wttv["wttv"].mean()

# %%
"""
# ATTV
"""
attv = nearby.metrics.attv(
    epochs["Target"],
    tmin=tmin,
    tmax=tmax,
    picks=["Cz", "CPz", "Pz"],
)

attv = attv["attv"].mean()

# %%
"""
# ATSVA
"""
atsva = nearby.metrics.atsv(
    epochs["Target"],
    tmin=tmin,
    tmax=tmax,
    metrics="angle",
)

atsva = atsva["atsva"].mean()

# %%
"""
Print results
"""

print("# Variability Metrics")
print(f" - ME    : {me:10.6f}")
print(f" - SMEAT : {smeat:10.6f}")
print(f" - WTTV  : {wttv:10.6f}")
print(f" - ATTV  : {attv:10.6f}")
print(f" - ATSVA : {atsva:10.6f}")
