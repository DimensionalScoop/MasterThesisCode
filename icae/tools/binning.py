import numpy as np
import pandas as pd

from icae.tools.preprocessing import quantile_partitions


def quantile_bin(
    df: pd.DataFrame, columns, count_bins, inplace=False, return_edges=False
):
    """
    Histogramizes the columns into discrete bins. Bin widths are chosen
    so that each bin has about the same amount of entries.
    :param df: DataFrame to work on
    :param columns: list of columns that should be binned
    :param count_bins: list of ints describing the number of bins to use
    for each column
    :param inplace: if true all manipulations are done on df, otherwise a copy
    is used
    :param return_edges: returns (df, bin edges) if true
    :return: df (inplace=False, return_edges=False), df, bin_edges (inplace=False, return_edges=True), None (inplace=True, return_edges=False), bin_edges (inplace=True, return_edges=True)
    """

    df = df if inplace else df.copy()
    bin_edges = []
    name_ext = "_quantile_binned"

    for col, bins in zip(columns, count_bins):
        binned, edges = quantile_partitions(df[col], bins)
        df[col + name_ext] = binned
        bin_edges.append(edges)

    prev_duplicates = np.sum(df.duplicated(subset=columns + ["frame"]))
    duplicates = np.sum(
        df.duplicated(subset=[col + name_ext for col in columns] + ["frame"])
    )
    print(
        "Duplicates introduced due to binning:",
        (duplicates - prev_duplicates) / len(df),
    )

    return_value = []
    if not inplace:
        return_value.append(df)
    if return_edges:
        return_value.append(bin_edges)
    if return_value:
        return return_value
    return None


def judge_binning_quality(quantiles, peak_region=(9000, 13500), unit="ns"):
    bin_sizes = quantiles[1:] - quantiles[:-1]

    print("Mean binning:", bin_sizes.mean(), unit)
    if peak_region is not None and len(peak_region) == 2:
        peak_region = (peak_region[0] < quantiles[:-1]) & (
            quantiles[:-1] < peak_region[1]
        )
        print("Peak binning:", bin_sizes[peak_region].mean(), unit)
    print("Worst 10% binning:", np.quantile(bin_sizes, 0.9), unit)
    print("Worst 1% binning:", np.quantile(bin_sizes, 0.99), unit)


def judge_frame_size(df: pd.DataFrame, columns, channels, print_info=True):
    dimensions = []
    for col in columns:
        dimensions.append(len(np.unique(df[col])))

    size_per_frame = np.prod(dimensions) * channels

    if print_info:
        print("Size per Frame", size_per_frame / 1e3 * 32 / 4, "KB")
        print("Total size", size_per_frame / 1e9 * 32 / 4 * df.frame.max(), "GB")

    return size_per_frame


def number_duplicates(
    df: pd.DataFrame, subset, levels, col_name="duplicity", return_info=False
):
    duplicity_after_numbering = []
    df.sort_values(by=subset, inplace=True)

    for dup in range(levels):
        if dup == 0:
            df[col_name] = 0
        duplicates = df.duplicated(subset=["frame", col_name] + subset)
        df.loc[duplicates, col_name] = dup + 1

        if return_info:
            duplicates = df.duplicated(subset=["frame", col_name] + subset)
            duplicity_after_numbering.append(duplicates.values.sum() / len(df))

    if return_info:
        return return_info

