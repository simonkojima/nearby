"""
Example: Variability of ERDS in a motor-imagery task
==================================================
"""

# %%
import mne
import pandas as pd
import nearby
from moabb.datasets import Dreyer2023

# %%
"""
Extract Epochs
"""

subject = 1
l_freq = 1
h_freq = 45
resample = 128
tmin = -2.0
tmax = 5.0

dataset = Dreyer2023()

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

mapping = dict()
for ch in raw.ch_names:
    if "EOG" in ch:
        mapping[ch] = "eog"
    elif "EMG" in ch:
        mapping[ch] = "emg"

raw.set_channel_types(mapping)

raw.pick(picks="eeg")

epochs = mne.Epochs(raw, baseline=None, tmin=tmin, tmax=tmax, event_repeated="merge")
epochs = epochs[["left_hand", "right_hand"]]

epochs.load_data()

epochs.resample(resample)

# %%
"""
Extract ERDS
"""

baseline = [-2.0, 0.0]

tfrs = epochs.compute_tfr(
    method="multitaper",
    freqs=list(range(l_freq, h_freq + 1)),
    n_cycles=list(range(l_freq, h_freq + 1)),
    use_fft=True,
    return_itc=False,
    average=False,
    decim=2,
    n_jobs=-1,
)

tfrs = tfrs.apply_baseline(baseline=baseline, mode="percent")

print(tfrs)

# %%
"""
Compute Variability Metrics

# ME
"""

fmin, fmax = 7, 13
fmin_atfv, fmax_atfv = 8, 30
tmin, tmax = 0.5, 4.5

me_left = nearby.metrics.me(
    tfrs["left_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C4",
    additional_values={"class": "left"},
)
me_right = nearby.metrics.me(
    tfrs["right_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C3",
    additional_values={"class": "right"},
)

me = pd.concat([me_left, me_right])["me"].mean()

# %%
"""
# SMEAT
"""
smeat_left = nearby.metrics.smeat(
    tfrs["left_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C4",
)
smeat_right = nearby.metrics.smeat(
    tfrs["right_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C3",
)

smeat = pd.concat([smeat_left, smeat_right])["smeat"].mean()

# %%
"""
# WTTV
"""
wttv_left = nearby.metrics.wttv(
    tfrs["left_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C4",
)
wttv_right = nearby.metrics.wttv(
    tfrs["right_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C3",
)

wttv = pd.concat([wttv_left, wttv_right])["wttv"].mean()

# %%
"""
# ATTV
"""
attv_left = nearby.metrics.attv(
    tfrs["left_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C4",
)
attv_right = nearby.metrics.attv(
    tfrs["right_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C3",
)

attv = pd.concat([attv_left, attv_right])["attv"].mean()

# %%
"""
# ATSVA
"""
atsva_left = nearby.metrics.atsv(
    tfrs["left_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    metrics="angle",
)
atsva_right = nearby.metrics.atsv(
    tfrs["right_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    metrics="angle",
)

atsva = pd.concat([atsva_left, atsva_right])["atsva"].mean()

# %%
"""
# ATFVA
"""
atfva_left = nearby.metrics.atfv(
    tfrs["left_hand"],
    fmin=fmin_atfv,
    fmax=fmax_atfv,
    tmin=tmin,
    tmax=tmax,
    metrics="angle",
)
atfva_right = nearby.metrics.atfv(
    tfrs["right_hand"],
    fmin=fmin_atfv,
    fmax=fmax_atfv,
    tmin=tmin,
    tmax=tmax,
    metrics="angle",
)

atfva = pd.concat([atfva_left, atfva_right])["atfva"].mean()

# %%
"""
# Print results
"""

print("# Variability Metrics")
print(f" - ME    : {me:10.6f}")
print(f" - SMEAT : {smeat:10.6f}")
print(f" - WTTV  : {wttv:10.6f}")
print(f" - ATTV  : {attv:10.6f}")
print(f" - ATSVA : {atsva:10.6f}")
print(f" - ATFVA : {atfva:10.6f}")
