import argparse
import pandas as pd
from pathlib import Path

pd.options.display.expand_frame_repr = False
pd.options.display.max_rows = None
pd.options.display.max_columns = None
pd.options.display.precision = 2

def plot_op_results(output_dir):
    assert output_dir.is_dir()
    files = output_dir.glob("*.csv")
    df = pd.concat(
        [pd.read_csv(f, sep="\t") for f in files], ignore_index=True, sort=False
    )
    print(
        df.pivot_table(
            index=["op", "nchannels"],
            columns="nthreads",
            values=["comp_time", "comm_time"],
            aggfunc="min",
            sort=False,
        ).reindex(["comp_time", "comm_time"], axis=1, level=0)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path, help="output directory")
    args = parser.parse_args()
    plot_op_results(args.output_dir)
