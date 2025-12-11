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
# Between-Average Temporal Variability (Mean Centered)
# ------------------------------------------

results = {}

tmin, tmax = 0.2, 0.4

evoked_list = list(evoked_dict.values())
tfrs_list = list(tfrs_dict.values())

evoked_M = mne.combine_evoked(evoked_list, weights="equal")
tfrs_M = mne.time_frequency.combine_tfr(tfrs_list, weights="equal")

# %%
# Between-Average Temporal Variability - Mean Centered (BATemp_MC)
# ----------------------------------------------------------------

ba_temp_mc = nearby.metrics.between_average_temporal(
    evoked_M,
    evoked_list,
    tmin=tmin,
    tmax=tmax,
    picks=["Cz", "CPz", "Pz"],
)

results["BATemp_MC"] = ba_temp_mc["between_average_temporal"].mean()
print(f"BATemp_MC: {results['BATemp_MC']:.3f}")

# %%
# Between-Average Spatial Variability - Mean Centered (BASpat_MC)
# ----------------------------------------------------------------

ba_spat_mc = nearby.metrics.between_average_spatial(
    evoked_M,
    evoked_list,
    tmin=tmin,
    tmax=tmax,
)

results["BASpat_MC"] = ba_spat_mc["between_average_spatial"].mean()
print(f"BASpat_MC: {results['BASpat_MC']:.3f}")

# %%
# Between-Average Frequency Variability - Mean Centered (BAFreq_MC)
# ----------------------------------------------------------------

ba_freq_mc = nearby.metrics.between_average_frequency(
    tfrs_M,
    tfrs_list,
    tmin=tmin,
    tmax=tmax,
)

results["BAFreq_MC"] = ba_freq_mc["between_average_frequency"].mean()
print(f"BAFreq_MC: {results['BAFreq_MC']:.3f}")

# %%
# Between-Average Temporal Variability (Test-User Centered)
# ====================================================
#

evoked_train_list = [evoked_dict[m] for m in train_subjects]
tfrs_train_list = [tfrs_dict[m] for m in train_subjects]

# %%
# Between-Average Temporal Variability - Test-User Centered (BATemp_TC)
# ---------------------------------------------------------------------

ba_temp_tc = nearby.metrics.between_average_temporal(
    evoked_dict[test_subject],
    evoked_train_list,
    tmin=tmin,
    tmax=tmax,
    picks=["Cz", "CPz", "Pz"],
)

results["BATemp_TC"] = ba_temp_tc["between_average_temporal"].mean()
print(f"BATemp_TC: {results['BATemp_TC']:.3f}")

# %%
# Between-Average Spatial Variability - Test-User Centered (BASpat_TC)
# ---------------------------------------------------------------------

ba_spat_tc = nearby.metrics.between_average_spatial(
    evoked_dict[test_subject],
    evoked_train_list,
    tmin=tmin,
    tmax=tmax,
)

results["BASpat_TC"] = ba_spat_tc["between_average_spatial"].mean()
print(f"BASpat_TC: {results['BASpat_TC']:.3f}")

# %%
# Between-Average Frequency Variability - Test-User Centered (BAFreq_TC)
# ---------------------------------------------------------------------

ba_freq_tc = nearby.metrics.between_average_frequency(
    tfrs_dict[test_subject],
    tfrs_train_list,
    tmin=tmin,
    tmax=tmax,
)

results["BAFreq_TC"] = ba_freq_tc["between_average_frequency"].mean()
print(f"BAFreq_TC: {results['BAFreq_TC']:.3f}")

# %%
# Results
# =======

print(pd.DataFrame(results, index=[0]))
