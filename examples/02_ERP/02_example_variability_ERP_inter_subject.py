"""
Example: Inter-Subject Variability of ERP in an auditory task
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

test_subject = 1
train_subjects = [2, 3, 4]
l_freq = 1
h_freq = 40
resample = 128
tmin_epochs = -0.1
tmax_epochs = 1.2
baseline = [-0.05, 0.0]

dataset = Kojima2024B(task="2stream")

subjects = [test_subject] + train_subjects

data = dataset.get_data(subjects=subjects)

evoked_dict = {}
tfrs_dict = {}

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

    evoked_dict[subject] = epochs.average()
    tfrs_dict[subject] = tfrs.average()

# %%
# Between-Group-Trial Variability
# ===============================

results = {}

tmin, tmax = 0.2, 0.4

evoked_list = list(evoked_dict.values())
tfrs_list = list(tfrs_dict.values())

evoked_M = mne.combine_evoked(evoked_list, weights="equal")
tfrs_M = mne.time_frequency.combine_tfr(tfrs_list, weights="equal")

# %%
# Between-Trial-Group Temporal Variability (BtwTrialGrpTemp)
# ----------------------------------------------------------

btw_trial_grp_temp = nearby.metrics.between_trial_group_temporal(
    evoked_list,
    tmin=tmin,
    tmax=tmax,
    picks=["Cz", "CPz", "Pz"],
)

results["BtwTrialGrpTemp"] = btw_trial_grp_temp["between_trial_group_temporal"].mean()
print(f"BtwTrialGrpTemp: {results['BtwTrialGrpTemp']:.3f}")

# %%
# Between-Trial-Group Spatial Variability (BtwTrialGrpSpat)
# ---------------------------------------------------------

btw_trial_grp_spat = nearby.metrics.between_trial_group_spatial(
    evoked_list,
    tmin=tmin,
    tmax=tmax,
    metric="angle",
)

results["BtwTrialGrpSpat"] = btw_trial_grp_spat["between_trial_group_spatial"].mean()
print(f"BtwTrialGrpSpat: {results['BtwTrialGrpSpat']:.3f}")

# %%
# Between-Trial-Group Frequency Variability (BtwTrialGrpFreq)
# -----------------------------------------------------------

btw_trial_grp_freq = nearby.metrics.between_trial_group_frequency(
    tfrs_list,
    tmin=tmin,
    tmax=tmax,
    metric="angle",
    picks=["Cz", "CPz", "Pz"],
)

results["BtwTrialGrpFreq"] = btw_trial_grp_freq["between_trial_group_frequency"].mean()
print(f"BtwTrialGrpFreq: {results['BtwTrialGrpFreq']:.3f}")

# %%
# Between-Trial-Group Variability (Test-User Referenced)
# ======================================================

evoked_train_list = [evoked_dict[m] for m in train_subjects]
tfrs_train_list = [tfrs_dict[m] for m in train_subjects]

# %%
# Between-Trial-Group Temporal Variability - Test-User Referenced (BtwTrialGrpTemp-TR)
# ------------------------------------------------------------------------------------

btw_trial_grp_temp_tr = nearby.metrics.between_trial_group_temporal(
    evoked_train_list,
    centroid=evoked_dict[test_subject],
    tmin=tmin,
    tmax=tmax,
    picks=["Cz", "CPz", "Pz"],
)

results["BtwTrialGrpTemp-TR"] = btw_trial_grp_temp_tr[
    "between_trial_group_temporal"
].mean()
print(f"BtwTrialGrpTemp-TR: {results['BtwTrialGrpTemp-TR']:.3f}")

# %%
# Between-Trial-Group Spatial Variability - Test-User Referenced (BtwTrialGrpSpat-TR)
# -----------------------------------------------------------------------------------

btw_trial_grp_spat_tr = nearby.metrics.between_trial_group_spatial(
    evoked_train_list,
    centroid=evoked_dict[test_subject],
    tmin=tmin,
    tmax=tmax,
    metric="angle",
)

results["BtwTrialGrpSpat-TR"] = btw_trial_grp_spat_tr[
    "between_trial_group_spatial"
].mean()
print(f"BtwTrialGrpSpat-TR: {results['BtwTrialGrpSpat-TR']:.3f}")

# %%
# Between-Trial-Group Frequency Variability - Test-User Referenced (BtwTrialGrpFreq-TR)
# -------------------------------------------------------------------------------------

btw_trial_grp_freq_tr = nearby.metrics.between_trial_group_frequency(
    tfrs_train_list,
    centroid=tfrs_dict[test_subject],
    tmin=tmin,
    tmax=tmax,
    metric="angle",
    picks=["Cz", "CPz", "Pz"],
)

results["BtwTrialGrpFreq-TR"] = btw_trial_grp_freq_tr[
    "between_trial_group_frequency"
].mean()
print(f"BtwTrialGrpFreq-TR: {results['BtwTrialGrpFreq-TR']:.3f}")

# %%
# Results
# =======

print(pd.DataFrame(results, index=[0]).to_string())
