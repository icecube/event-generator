import os
import glob
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def warning(msg):
    print(bcolors.WARNING + msg + bcolors.ENDC)


def error(msg):
    print(bcolors.FAIL + msg + bcolors.ENDC)


files = [
    "NuGen/NuE/low_energy/IC86_2013_holeice_30_v4/l5/1/DNN_l5_00000002.hdf5",
    "NuGen/NuE/medium_energy/IC86_2013_holeice_30_v4/l5/1/DNN_l5_00000001.hdf5",
]

keys_warning = [
    # sanity checks
    "I3EventHeader",
    # ic3-labels labels
    "LabelsDeepLearning",
    "LabelsMCCascade",
    "MCCascade",
    # covariance matrices are very sensitive to small changes and thus not
    # always reproducible
    "EventGenerator_cascade_7param_noise_ftpv3m__big_01_cov_matrix_cov",
    "EventGenerator_cascade_7param_noise_ftpv3m__big_01_cov_matrix_cov_fit",
    "EventGenerator_cascade_7param_noise_ftpv3m__big_01_cov_matrix_cov_fit_trafo",
    "EventGenerator_cascade_7param_noise_ftpv3m__big_01_cov_matrix_cov_sand",
    "EventGenerator_cascade_7param_noise_ftpv3m__big_01_cov_matrix_cov_sand_fit",
    "EventGenerator_cascade_7param_noise_ftpv3m__big_01_cov_matrix_cov_sand_fit_trafo",
    "EventGenerator_cascade_7param_noise_ftpv3m__big_01_cov_matrix_cov_sand_trafo",
    "EventGenerator_cascade_7param_noise_ftpv3m__big_01_cov_matrix_cov_trafo",
]

keys_error = [
    # event-generator results
    "EventGenerator_cascade_7param_noise_ftpv3m__big_01",
    "EventGenerator_cascade_7param_noise_ftpv3m__big_01_I3Particle",
]

dir_original = os.path.join(
    SCRIPT_DIR, "test_data/egenerator_test_01_base_v2_0_0"
)
test_dirs = glob.glob(os.path.join(SCRIPT_DIR, "test_data/*"))
test_dirs.remove(dir_original)

if len(test_dirs) == 0:
    raise ValueError("No test directories found!")

warnings = []
got_warning = False
passed_test = True
for dir_test in test_dirs:
    print("\nNow testing {!r} against {!r}".format(dir_test, dir_original))
    for file_name in files:
        print("\n\tNow testing {!r}".format(file_name))
        for key in keys_warning + keys_error:
            try:
                df_original = pd.read_hdf(
                    os.path.join(dir_original, file_name), key=key
                )
                df_test = pd.read_hdf(
                    os.path.join(dir_test, file_name), key=key
                )
            except Exception as e:
                warning("\t\tProblem with key {!r}".format(key))
                warning("\t\t", e)
                got_warning = True

            if "minimization_success" in df_original.columns:
                minimization_success = np.logical_and(
                    df_test["minimization_success"],
                    df_original["minimization_success"],
                )
            else:
                minimization_success = np.ones(len(df_test), dtype=bool)

            assert (df_original.columns == df_test.columns).all()
            for k in df_original.columns:

                # if all values are NaN, skip
                if (
                    np.isnan(df_original[k].values).all()
                    and np.isnan(df_test[k].values).all()
                ):
                    continue

                if "runtime" not in k:

                    # set tolerances
                    atol = 5e-6
                    rtol = 5e-4
                    rtol_fatal = 0.3

                    if not np.allclose(
                        df_original[k].values,
                        df_test[k].values,
                        atol=atol,
                        rtol=rtol,
                    ):
                        # compute relative difference
                        diff = df_original[k].values - df_test[k].values
                        rel_diff = diff / np.abs(df_original[k].values)
                        rel_diff_max = np.max(np.abs(rel_diff))
                        warnings.append(
                            [
                                key,
                                k,
                                rel_diff_max,
                                rel_diff_max > rtol_fatal
                                and key in keys_error,
                            ]
                        )

                        mask = np.abs(rel_diff) > rtol

                        if np.all(~minimization_success[mask]):
                            min_errors = True
                        else:
                            min_errors = False

                        if key in keys_warning:
                            warning("\t\tWarning: mismatch for {}".format(k))
                            got_warning = True
                        elif key in keys_error:
                            if rel_diff_max > rtol_fatal and not min_errors:
                                passed_test = False
                                error("\t\tError: mismatch for {}".format(k))
                            else:
                                warning(
                                    "\t\tWarning: mismatch for {}".format(k)
                                )
                                got_warning = True
                        else:
                            raise KeyError("Unknown key {!r}".format(key))
                        print(f"\t\tKey: {key} | column: {k}")
                        print("\t\tElement-wise difference:")
                        print("\t\t", diff[mask])
                        print("\t\tRelative difference:")
                        print("\t\t", rel_diff[mask])
                        print("\t\tOriginal:")
                        print("\t\t", df_original[k].values[mask])
                        print("\t\tTest:")
                        print("\t\t", df_test[k].values[mask])
                else:
                    runtime_orig = np.mean(df_original[k].values) * 1000.0
                    runtime_orig_std = np.std(df_original[k].values) * 1000.0
                    runtime_test = np.mean(df_test[k].values) * 1000.0
                    runtime_test_std = np.std(df_test[k].values) * 1000.0
                    max_dev = max(2 * runtime_orig_std, 0.5 * runtime_orig)
                    if np.abs(runtime_orig - runtime_test) > max_dev:
                        msg = "\t\t Runtimes: {:3.3f} +- {:3.3f}ms [base] "
                        msg += "{:3.3f} +- {:3.3f}ms [test]"
                        print(
                            msg.format(
                                runtime_orig,
                                runtime_orig_std,
                                runtime_test,
                                runtime_test_std,
                            )
                        )

# print warnings
if len(warnings) > 0:
    max_chars = 25
    print(f"\n{'Rel. diff.':8s} | {'Key':25s} | Column")
    print("=" * (max_chars * 2 + 16))
    for key, k, max_rel_diff, fatal in warnings:
        if len(k) > max_chars:
            k = k[:3] + "..." + k[-(max_chars - 6) :]
        if len(key) > max_chars:
            key = key[:3] + "..." + key[-(max_chars - 6) :]

        msg = f"{max_rel_diff*100.:9.3f}% | {key:25s} | {k}"
        if fatal:
            print(bcolors.FAIL + msg + bcolors.ENDC)
        else:
            print(bcolors.WARNING + msg + bcolors.ENDC)

print("\n====================")
print("=== Summary ========")
print("====================")
if got_warning:
    print(
        "=== "
        + bcolors.WARNING
        + "Warnings: {}".format(got_warning)
        + bcolors.ENDC
    )
else:
    print(
        "==="
        + bcolors.OKGREEN
        + " Warnings: {}".format(got_warning)
        + bcolors.ENDC
    )
if passed_test:
    print(
        "==="
        + bcolors.OKGREEN
        + " Passed:   {}".format(passed_test)
        + bcolors.ENDC
    )
else:
    print(
        "==="
        + bcolors.FAIL
        + " Passed:   {}".format(passed_test)
        + bcolors.ENDC
    )
print("====================\n")
