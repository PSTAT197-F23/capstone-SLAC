# coding: utf-8
import pandas
import sys
from scipy.stats import linregress, pearsonr
import pylab as plt
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("csv", type=str, help="Input CSV file from Jade's script")
parser.add_argument("--out", type=str, help="Output image file name")
args = parser.parse_args()

df = pandas.read_csv(args.csv)
fig, axs = plt.subplots(nrows=1, ncols=2)

for ax, coord in zip(axs, ["X", "Y"]):
    
    col, col_gt = "%s Prediction" % coord, "%s Actual" % coord

    c = pearsonr(df[col], df[col_gt])[0]
    ax.plot( df[col], df[col_gt], '.')
    ax.set_xlabel(col, labelpad=0)
    ax.set_ylabel(col_gt, labelpad=0)
    ax.set_title("C=%.4f" % c, pad=0)
    ax.tick_params(pad=1, length=0)
    ax.grid(1, ls="--", alpha=0.5)

plt.suptitle("%s" % args.csv, fontsize=8)
fig.set_size_inches((5,2.5))
plt.subplots_adjust(left=0.15, bottom=0.17, right=0.95, top=0.80, wspace=0.45)
if args.out is not None:
    plt.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}.")
plt.show()

