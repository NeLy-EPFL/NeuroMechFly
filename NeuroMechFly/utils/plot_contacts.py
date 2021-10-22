#!/usr/bin/env python

""" Function to plot contact information """
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rc_params = {
    "axes.labelsize": 10,
    'font.size': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'savefig.dpi': 300,
    'savefig.format': 'png'
}
plt.rcParams.update(rc_params)


def parse_args():
    """ Parse arguments"""
    parser = ArgumentParser("Plot contacts")
    parser.add_argument(
        "--file_path", "-f", required=True, dest="file_path", type=str,
        help="File path pointing to contacts.h5 file"
    )
    parser.add_argument(
        "--output_path", "-o", required=True, dest="output_path", type=str,
        help="Path to export the figure to"
    )
    return parser.parse_args()


def plot_gait_diagram(data, ts=1e-4, ax=None):
    """ Plot the contacts from the given data """
    # Total time
    total_time = len(data)*ts
    # Define the legs and its order for the plot
    legs = ("RH", "RM", "RF", "LH", "LM", "LF")
    # Setup the contact data
    contact_intervals = {}
    for leg in legs:
        # Combine contact information of all the tarsis segments
        values = np.squeeze(np.any(
            [value for key, value in data.items() if leg in key],
            axis=0,
        ).astype(int))
        intervals = np.where(
            np.abs(np.diff(values, prepend=[0], append=[0])) == 1
        )[0].reshape(-1, 2)*ts
        intervals[:, 1] = intervals[:, 1] - intervals[:, 0]
        contact_intervals[leg] = intervals
    # Define the figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))
    width = 0.85
    for index, (key, value) in enumerate(contact_intervals.items()):
        ax.broken_barh(
            value, (index-width*0.5, width), facecolor='k'
        )
    ax.set_xlabel("Time [s]")
    ax.set_yticks((0, 1, 2, 3, 4, 5))
    ax.set_yticklabels(legs)


def main():
    """ Main function """
    # CLI args
    args = parse_args()
    # Load the data
    data = pd.read_hdf(args.file_path)
    # Plot and export the figure
    plot(data, args.output_path)


if __name__ == '__main__':
    main()
