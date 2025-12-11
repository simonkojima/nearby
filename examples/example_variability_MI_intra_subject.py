"""
Example: Intra-Subject Variability of ERDS in a motor-imagery task
==================================================================
"""

# Authors: Simon Kojima <simon.kojima@inria.fr>
#
# License: BSD (3-clause)

import mne
import pandas as pd
import nearby
from moabb.datasets import Dreyer2023

# %%
# Extract Epochs
# ==============

subject = 1
l_freq = 7
h_freq = 13
resample = 128
tmin_epochs = -2.5
tmax_epochs = 5.5

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

raw.pick(picks="eeg")

epochs = mne.Epochs(
    raw, baseline=None, tmin=tmin_epochs, tmax=tmax_epochs, event_repeated="merge"
)
epochs = epochs[["left_hand", "right_hand"]]

epochs.load_data()

epochs.resample(resample)

# %%
# Extract ERDS
# ============

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

# %%
# Within-Trial Variability Metrics
# ================================
#

results = {}

fmin, fmax = 7, 13
tmin, tmax = dataset.interval[0] + 0.5, dataset.interval[1]

# %%
# Within-Trial Temporal Variability (WTTemp)
# ------------------------------------------

wt_temp_left = nearby.metrics.within_trial_temporal(
    tfrs["left_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C4",
)
wt_temp_right = nearby.metrics.within_trial_temporal(
    tfrs["right_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C3",
)
wt_temp = pd.concat([wt_temp_left, wt_temp_right])
results["WTTemp"] = wt_temp["within_trial_temporal"].mean()
print(f"WTTemp: {results['WTTemp']:.3f}")

# %%
# Within-Trial Spatial Variability (WTSpat)
# ------------------------------------------

wt_spat_left = nearby.metrics.within_trial_spatial(
    tfrs["left_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    auto_window_size=1,
    auto_window_step=1,
    metric="angle",
)
wt_spat_right = nearby.metrics.within_trial_spatial(
    tfrs["right_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    auto_window_size=1,
    auto_window_step=1,
    metric="angle",
)
wt_spat = pd.concat([wt_spat_left, wt_spat_right])
results["WTSpat"] = wt_spat["within_trial_spatial"].mean()
print(f"WTSpat: {results['WTSpat']:.3f}")

# %%
# Within-Trial Frequency Variability (WTFreq)
# -------------------------------------------
wt_freq_left = nearby.metrics.within_trial_frequency(
    tfrs["left_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    auto_window_size=1,
    auto_window_step=1,
    metric="angle",
    picks="C4",
)

wt_freq_right = nearby.metrics.within_trial_frequency(
    tfrs["right_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    auto_window_size=1,
    auto_window_step=1,
    metric="angle",
    picks="C3",
)

wt_freq = pd.concat([wt_freq_left, wt_freq_right])
results["WTFreq"] = wt_freq["within_trial_frequency"].mean()
print(f"WTFreq: {results['WTFreq']:.3f}")

# %%
# Between-Trial Variability
# =========================


# %%
# Between-Trial Spatial Variability
# ----------------------------------

bt_temp_left = nearby.metrics.between_trial_temporal(
    tfrs["left_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C4",
)
bt_temp_right = nearby.metrics.between_trial_temporal(
    tfrs["right_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C3",
)

bt_temp = pd.concat([bt_temp_left, bt_temp_right])
results["BTTemp"] = bt_temp["between_trial_temporal"].mean()
print(f"BTTemp: {results['BTTemp']:.3f}")

# %%
# Between-Trial Spatial Variability
# ----------------------------------

bt_spat_left = nearby.metrics.between_trial_spatial(
    tfrs["left_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
)
bt_spat_right = nearby.metrics.between_trial_spatial(
    tfrs["right_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
)

bt_spat = pd.concat([bt_spat_left, bt_spat_right])
results["BTSpat"] = bt_spat["between_trial_spatial"].mean()
print(f"BTSpat: {results['BTSpat']:.3f}")

# %%
# Between-Trial Frequency Variability
# ----------------------------------

bt_freq_left = nearby.metrics.between_trial_frequency(
    tfrs["left_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C4",
)
bt_freq_right = nearby.metrics.between_trial_frequency(
    tfrs["right_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C3",
)

bt_freq = pd.concat([bt_freq_left, bt_freq_right])
results["BTFreq"] = bt_freq["between_trial_frequency"].mean()
print(f"BTFreq: {results['BTFreq']:.3f}")

# %%
# Results
# =======

print(pd.DataFrame(results, index=[0]))
