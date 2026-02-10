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
# Between-Group-Trial Variability
# ===============================

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
# Between-Trial-Group Temporal Variability (BtwTrialGrpTemp)
# ----------------------------------------------------------

import importlib

importlib.reload(nearby)
importlib.reload(nearby.metrics)

btw_trial_grp_temp_left = nearby.metrics.between_trial_group_temporal(
    tfrs_left,
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C4",
)

btw_trial_grp_temp_right = nearby.metrics.between_trial_group_temporal(
    tfrs_right,
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C3",
)

btw_trial_grp_temp = pd.concat([btw_trial_grp_temp_left, btw_trial_grp_temp_right])
results["BtwTrialGrpTemp"] = btw_trial_grp_temp["between_trial_group_temporal"].mean()
print(f"BtwTrialGrpTemp: {results['BtwTrialGrpTemp']:.3f}")

# %%
# Between-Trial-Group Spatial Variability (BtwTrialGrpSpat)
# ---------------------------------------------------------

btw_trial_grp_spat_left = nearby.metrics.between_trial_group_spatial(
    tfrs_left, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, metric="angle"
)

btw_trial_grp_spat_right = nearby.metrics.between_trial_group_spatial(
    tfrs_right, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, metric="angle"
)

btw_trial_grp_spat = pd.concat([btw_trial_grp_spat_left, btw_trial_grp_spat_right])
results["BtwTrialGrpSpat"] = btw_trial_grp_spat["between_trial_group_spatial"].mean()
print(f"BtwTrialGrpSpat: {results['BtwTrialGrpSpat']:.3f}")

# %%
# Between-Trial-Group Frequency Variability (BtwTrialGrpFreq)
# -----------------------------------------------------------

btw_trial_grp_freq_left = nearby.metrics.between_trial_group_frequency(
    tfrs_left,
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    metric="angle",
    picks="C4",
)

btw_trial_grp_freq_right = nearby.metrics.between_trial_group_frequency(
    tfrs_right,
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    metric="angle",
    picks="C3",
)

btw_trial_grp_freq = pd.concat([btw_trial_grp_freq_left, btw_trial_grp_freq_right])
results["BtwTrialGrpFreq"] = btw_trial_grp_freq["between_trial_group_frequency"].mean()
print(f"BtwTrialGrpFreq: {results['BtwTrialGrpFreq']:.3f}")

# %%
# Between-Trial-Group Variability (Test-User Referenced)
# ======================================================

tfrs_train_left = [tfrs_dict["left_hand"][m] for m in train_subjects]
tfrs_train_right = [tfrs_dict["right_hand"][m] for m in train_subjects]

# %%
# Between-Trial-Group Temporal Variability - Test-User Referenced (BtwTrialGrpTemp-TR)
# ------------------------------------------------------------------------------------

btw_trial_grp_temp_tr_left = nearby.metrics.between_trial_group_temporal(
    tfrs_train_left,
    centroid=tfrs_dict["left_hand"][test_subject],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C4",
)

btw_trial_grp_temp_tr_right = nearby.metrics.between_trial_group_temporal(
    tfrs_train_right,
    centroid=tfrs_dict["right_hand"][test_subject],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    picks="C3",
)

btw_trial_grp_temp_tr = pd.concat(
    [btw_trial_grp_temp_tr_left, btw_trial_grp_temp_tr_right]
)
results["BtwTrialGrpTemp-TR"] = btw_trial_grp_temp_tr[
    "between_trial_group_temporal"
].mean()
print(f"BtwTrialGrpTemp-TR: {results['BtwTrialGrpTemp-TR']:.3f}")

# %%
# Between-Trial-Group Spatial Variability - Test-User Referenced (BtwTrialGrpSpat-TR)
# -----------------------------------------------------------------------------------

btw_trial_grp_spat_tr_left = nearby.metrics.between_trial_group_spatial(
    tfrs_train_left,
    centroid=tfrs_dict["left_hand"][test_subject],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    metric="angle",
)

btw_trial_grp_spat_tr_right = nearby.metrics.between_trial_group_spatial(
    tfrs_train_right,
    centroid=tfrs_dict["right_hand"][test_subject],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    metric="angle",
)

btw_trial_grp_spat_tr = pd.concat(
    [btw_trial_grp_spat_tr_left, btw_trial_grp_spat_tr_right]
)
results["BtwTrialGrpSpat-TR"] = btw_trial_grp_spat_tr[
    "between_trial_group_spatial"
].mean()
print(f"BtwTrialGrpSpat-TR: {results['BtwTrialGrpSpat-TR']:.3f}")

# %%
# Between-Trial-Group Frequency Variability - Test-User Referenced (BtwTrialGrpFreq-TR)
# -------------------------------------------------------------------------------------

btw_trial_grp_freq_tr_left = nearby.metrics.between_trial_group_frequency(
    tfrs_train_left,
    centroid=tfrs_dict["left_hand"][test_subject],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    metric="angle",
)

btw_trial_grp_freq_tr_right = nearby.metrics.between_trial_group_frequency(
    tfrs_train_right,
    centroid=tfrs_dict["right_hand"][test_subject],
    fmin=fmin,
    fmax=fmax,
    tmin=tmin,
    tmax=tmax,
    metric="angle",
)

btw_trial_grp_freq_tr = pd.concat(
    [btw_trial_grp_freq_tr_left, btw_trial_grp_freq_tr_right]
)
results["BtwTrialGrpFreq-TR"] = btw_trial_grp_freq_tr[
    "between_trial_group_frequency"
].mean()
print(f"BtwTrialGrpFreq-TR: {results['BtwTrialGrpFreq-TR']:.3f}")

# %%
# Results
# =======

print(pd.DataFrame(results, index=[0]).to_string())
