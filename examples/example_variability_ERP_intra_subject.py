"""
Example: Intra-Subject Variability of ERP in an auditory task
=============================================================
"""

# Authors: Simon Kojima <simon.kojima@inria.fr>
#
# License: BSD (3-clause)

import mne
import nearby
from moabb.datasets import Kojima2024B
import pandas as pd

# %%
# Extract Epochs
# ==============

subject = 1
l_freq = 1
h_freq = 40
resample = 128
tmin_epochs = -0.1
tmax_epochs = 1.2
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

epochs = mne.Epochs(raw, baseline=None, tmin=tmin_epochs, tmax=tmax_epochs)

epochs.load_data()

epochs.resample(resample)

epochs = epochs.apply_baseline(baseline=baseline)

epochs = epochs["Target"]

tfrs = epochs.compute_tfr(
    method="multitaper",
    freqs=list(range(l_freq, h_freq + 1, 3)),
    n_cycles=list(range(l_freq, h_freq + 1, 3)),
    use_fft=True,
    return_itc=False,
    average=False,
    decim=2,
    n_jobs=-1,
)

tfrs = tfrs.apply_baseline(baseline=baseline, mode="percent")

# %%
# Within-Trial Variability Metrics
# ================================

results = {}

tmin, tmax = 0.2, 0.4

# %%
# Within-Trial Temporal Variability (WTTemp)
# ------------------------------------------

wt_temp = nearby.metrics.within_trial_temporal(
    epochs["Target"],
    tmin=tmin,
    tmax=tmax,
    picks=["Cz", "CPz", "Pz"],
)

results["WTTemp"] = wt_temp["within_trial_temporal"].mean()
print(f"WTTemp: {results['WTTemp']:.3f}")

# %%
# Within-Trial Spatial Variability (WTSpat)
# ------------------------------------------

wt_spat = nearby.metrics.within_trial_spatial(
    epochs["Target"],
    tmin=tmin,
    tmax=tmax,
    auto_window_size=0.05,
    auto_window_step=0.05,
)

results["WTSpat"] = wt_spat["within_trial_spatial"].mean()
print(f"WTSpat: {results['WTSpat']:.3f}")

# %%
# Within-Trial Frequency Variability (WTFreq)
# ------------------------------------------

wt_spat = nearby.metrics.within_trial_frequency(
    tfrs["Target"],
    tmin=tmin,
    tmax=tmax,
    auto_window_size=0.05,
    auto_window_step=0.05,
    picks=["Cz", "CPz", "Pz"],
)

results["WTFreq"] = wt_spat["within_trial_frequency"].mean()
print(f"WTFreq: {results['WTFreq']:.3f}")

# %%
# Between-Trial Variability Metrics
# ================================

# %%
# Between-Trial Temporal Variability (BTTemp)
# ------------------------------------------

bt_temp = nearby.metrics.between_trial_temporal(
    epochs["Target"],
    tmin=tmin,
    tmax=tmax,
    picks=["Cz", "CPz", "Pz"],
)

results["BTTemp"] = bt_temp["between_trial_temporal"].mean()
print(f"BTTemp: {results['BTTemp']:.3f}")

# %%
# Between-Trial Spatial Variability (BTSpat)
# ------------------------------------------

bt_spat = nearby.metrics.between_trial_spatial(
    epochs["Target"],
    tmin=tmin,
    tmax=tmax,
)

results["BTSpat"] = bt_spat["between_trial_spatial"].mean()
print(f"BTSpat: {results['BTSpat']:.3f}")

# %%
# Between-Trial Frequency Variability (BTFreq)
# ------------------------------------------

bt_freq = nearby.metrics.between_trial_frequency(
    tfrs["Target"],
    tmin=tmin,
    tmax=tmax,
    picks=["Cz", "CPz", "Pz"],
)

results["BTFreq"] = bt_freq["between_trial_frequency"].mean()
print(f"BTFreq: {results['BTFreq']:.3f}")

# %%
# Results
# =======

print(pd.DataFrame(results, index=[0]))
