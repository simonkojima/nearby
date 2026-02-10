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
# Within-Trial Temporal Variability (WiTrialTemp)
# -----------------------------------------------

wi_trial_temp_left = nearby.metrics.within_trial_temporal(
    tfrs["left_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C4",
)
wi_trial_temp_right = nearby.metrics.within_trial_temporal(
    tfrs["right_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C3",
)
wi_trial_temp = pd.concat([wi_trial_temp_left, wi_trial_temp_right])
results["WiTrialTemp"] = wi_trial_temp["within_trial_temporal"].mean()
print(f"WiTrialTemp: {results['WiTrialTemp']:.3f}")

# %%
# Within-Trial Spatial Variability (WiTrialSpat)
# ----------------------------------------------

wi_trial_spat_left = nearby.metrics.within_trial_spatial(
    tfrs["left_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    auto_window_size=1,
    auto_window_step=1,
    metric="angle",
)
wi_trial_spat_right = nearby.metrics.within_trial_spatial(
    tfrs["right_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    auto_window_size=1,
    auto_window_step=1,
    metric="angle",
)
wi_trial_spat = pd.concat([wi_trial_spat_left, wi_trial_spat_right])
results["WiTrialSpat"] = wi_trial_spat["within_trial_spatial"].mean()
print(f"WiTrialSpat: {results['WiTrialSpat']:.3f}")

# %%
# Within-Trial Frequency Variability (WiTrialFreq)
# ------------------------------------------------
wi_trial_freq_left = nearby.metrics.within_trial_frequency(
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

wi_trial_freq_right = nearby.metrics.within_trial_frequency(
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

wi_trial_freq = pd.concat([wi_trial_freq_left, wi_trial_freq_right])
results["WiTrialFreq"] = wi_trial_freq["within_trial_frequency"].mean()
print(f"WiTrialFreq: {results['WiTrialFreq']:.3f}")

# %%
# Between-Trial Variability
# =========================


# %%
# Between-Trial Temporal Variability (BtwTrialTemp)
# -------------------------------------------------

btw_trial_temp_left = nearby.metrics.between_trial_temporal(
    tfrs["left_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C4",
)
btw_trial_temp_right = nearby.metrics.between_trial_temporal(
    tfrs["right_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C3",
)

btw_trial_temp = pd.concat([btw_trial_temp_left, btw_trial_temp_right])
results["BtwTrialTemp"] = btw_trial_temp["between_trial_temporal"].mean()
print(f"BtwTrialTemp: {results['BtwTrialTemp']:.3f}")

# %%
# Between-Trial Spatial Variability (BtwTrialSpat)
# ------------------------------------------------

btw_trial_spat_left = nearby.metrics.between_trial_spatial(
    tfrs["left_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    metric="angle",
)
btw_trial_spat_right = nearby.metrics.between_trial_spatial(
    tfrs["right_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    metric="angle",
)

btw_trial_spat = pd.concat([btw_trial_spat_left, btw_trial_spat_right])
results["BtwTrialSpat"] = btw_trial_spat["between_trial_spatial"].mean()
print(f"BtwTrialSpat: {results['BtwTrialSpat']:.3f}")

# %%
# Between-Trial Frequency Variability (BtwTrialFreq)
# --------------------------------------------------

btw_trial_freq_left = nearby.metrics.between_trial_frequency(
    tfrs["left_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    metric="angle",
    picks="C4",
)
btw_trial_freq_right = nearby.metrics.between_trial_frequency(
    tfrs["right_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    metric="angle",
    picks="C3",
)

btw_trial_freq = pd.concat([btw_trial_freq_left, btw_trial_freq_right])
results["BtwTrialFreq"] = btw_trial_freq["between_trial_frequency"].mean()
print(f"BtwTrialFreq: {results['BtwTrialFreq']:.3f}")

# %%
# Class Stability
# ===============

class_stability_left = nearby.metrics.class_stability(
    epochs=epochs["left_hand"], tmin=tmin, tmax=tmax
)
class_stability_right = nearby.metrics.class_stability(
    epochs=epochs["right_hand"], tmin=tmin, tmax=tmax
)

class_stability = pd.concat([class_stability_left, class_stability_right])
results["class_stability"] = class_stability["class_stability"].mean()
print(f"class_stability: {results['class_stability']:.3f}")

# %%
# STDERD
# ------

std_erd_left = nearby.metrics.standard_deviation_erds(
    epochs["left_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C4",
)
std_erd_right = nearby.metrics.standard_deviation_erds(
    epochs["right_hand"],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C3",
)
std_erd = pd.concat([std_erd_left, std_erd_right])
results["STDERD"] = std_erd["standard_deviation_erds"].mean()
print(f"STDERD: {results['STDERD']:.3f}")

# %%
# Results
# =======

print(pd.DataFrame(results, index=[0]).to_string())
