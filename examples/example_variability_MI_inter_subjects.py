"""
Example: Inter-Subjects Variability of ERDS in a motor-imagery task
===================================================================
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

test_subject = 1
train_subjects = [2, 3, 4]
l_freq = 7
h_freq = 13
resample = 128
tmin_epochs = -2.5
tmax_epochs = 5.5

dataset = Dreyer2023()

subjects = [test_subject] + train_subjects
data = dataset.get_data(subjects=subjects)

epochs = {}

for subject in subjects:

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

    e = mne.Epochs(
        raw, baseline=None, tmin=tmin_epochs, tmax=tmax_epochs, event_repeated="merge"
    )
    e = e[["left_hand", "right_hand"]]

    e.load_data()

    e.resample(resample)

    epochs[subject] = e

# %%
# Extract ERDS
# ============

baseline = [-2.0, 0.0]

tfrs_dict = {"left_hand": {}, "right_hand": {}}

for subject in subjects:
    tfrs = epochs[subject].compute_tfr(
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

    tfrs_dict["left_hand"][subject] = tfrs["left_hand"].average()
    tfrs_dict["right_hand"][subject] = tfrs["right_hand"].average()

# %%
# Between-Average Temporal Variability (Mean Centered)
# ====================================================
#

results = {}

fmin, fmax = 7, 13
tmin, tmax = dataset.interval[0] + 0.5, dataset.interval[1]

tfrs_left = list(tfrs_dict["left_hand"].values())
tfrs_right = list(tfrs_dict["right_hand"].values())

M_left = mne.time_frequency.combine_tfr(
    tfrs_left,
    weights="equal",
)
M_right = mne.time_frequency.combine_tfr(
    tfrs_right,
    weights="equal",
)

# %%
# Between-Average Temporal Variability - Mean Centered (BATemp_MC)
# ------------------------------------------

ba_temp_mc_left = nearby.metrics.between_average_temporal(
    M_left,
    tfrs_left,
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C4",
)

ba_temp_mc_right = nearby.metrics.between_average_temporal(
    M_right,
    tfrs_right,
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C3",
)

ba_temp_mc = pd.concat([ba_temp_mc_left, ba_temp_mc_right])
results["BATemp_MC"] = ba_temp_mc["between_average_temporal"].mean()
print(f"BATemp_MC: {results['BATemp_MC']:.3f}")

# %%
# Between-Average Spatial Variability - Mean Centered (BASpat_MC)
# ------------------------------------------

ba_spat_mc_left = nearby.metrics.between_average_spatial(
    M_left,
    tfrs_left,
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
)

ba_spat_mc_right = nearby.metrics.between_average_spatial(
    M_right,
    tfrs_right,
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
)

ba_spat_mc = pd.concat([ba_spat_mc_left, ba_spat_mc_right])
results["BASpat_MC"] = ba_spat_mc["between_average_spatial"].mean()
print(f"BASpat_MC: {results['BASpat_MC']:.3f}")

# %%
# Between-Average Frequency Variability - Mean Centered (BAFreq_MC)
# ------------------------------------------

ba_freq_mc_left = nearby.metrics.between_average_frequency(
    M_left,
    tfrs_left,
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C4",
)

ba_freq_mc_right = nearby.metrics.between_average_frequency(
    M_right,
    tfrs_right,
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C3",
)

ba_freq_mc = pd.concat([ba_freq_mc_left, ba_freq_mc_right])
results["BAFreq_MC"] = ba_freq_mc["between_average_frequency"].mean()
print(f"BAFreq_MC: {results['BAFreq_MC']:.3f}")

# %%
# Between-Average Temporal Variability (Test-User Centered)
# ====================================================
#

tfrs_train_left = [tfrs_dict["left_hand"][m] for m in train_subjects]
tfrs_train_right = [tfrs_dict["right_hand"][m] for m in train_subjects]

# %%
# Between-Average Temporal Variability - Test-User Centered (BATemp_TC)
# ------------------------------------------

ba_temp_tc_left = nearby.metrics.between_average_temporal(
    tfrs_dict["left_hand"][test_subject],
    tfrs_train_left,
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C4",
)

ba_temp_tc_right = nearby.metrics.between_average_temporal(
    tfrs_dict["right_hand"][test_subject],
    tfrs_train_right,
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C3",
)

ba_temp_tc = pd.concat([ba_temp_tc_left, ba_temp_tc_right])
results["BATemp_TC"] = ba_temp_tc["between_average_temporal"].mean()
print(f"BATemp_TC: {results['BATemp_TC']:.3f}")

# %%
# Between-Average Spatial Variability - Test-User Centered (BASpat_TC)
# ------------------------------------------

ba_spat_tc_left = nearby.metrics.between_average_spatial(
    tfrs_dict["left_hand"][test_subject],
    tfrs_train_left,
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
)

ba_spat_tc_right = nearby.metrics.between_average_spatial(
    tfrs_dict["right_hand"][test_subject],
    tfrs_train_right,
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
)

ba_spat_tc = pd.concat([ba_spat_tc_left, ba_spat_tc_right])
results["BASpat_TC"] = ba_spat_tc["between_average_spatial"].mean()
print(f"BASpat_TC: {results['BASpat_TC']:.3f}")

# %%
# Between-Average Frequency Variability - Test-User Centered (BAFreq_TC)
# ------------------------------------------

ba_freq_tc_left = nearby.metrics.between_average_frequency(
    tfrs_dict["left_hand"][test_subject],
    tfrs_train_left,
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
)

ba_freq_tc_right = nearby.metrics.between_average_frequency(
    tfrs_dict["right_hand"][test_subject],
    tfrs_train_right,
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
)

ba_freq_tc = pd.concat([ba_freq_tc_left, ba_freq_tc_right])
results["BAFreq_TC"] = ba_freq_tc["between_average_frequency"].mean()
print(f"BAFreq_TC: {results['BAFreq_TC']:.3f}")

# %%
# Results
# =======

print(pd.DataFrame(results, index=[0]))
