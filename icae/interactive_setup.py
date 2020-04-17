try:
    from IPython import get_ipython

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
    # get_ipython().run_line_magic('matplotlib', 'inline')
    print("Auto-reloading enabled.")
except AttributeError:
    pass

import sys

# XXX: is this necessary?
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm, colors
import sys, os
import errno
import re
import datetime
from icae.tools.config_loader import config
import box


def mkdir_p(path):
    """python2 way of running 'mkdir -p [path]'"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def set_plot_path(__file__):
    global plot_path
    plot_path = __file__
    try:
        plt.plot([1, 2, 3])
        plt.show(block=False)
        plt.clf()
    except:
        print("Problems with plotting, trying again with different backend…")
        matplotlib.use("agg")
        plt.plot([1, 2, 3])
        plt.show(block=False)
        plt.clf()


plot_path = None
disable_plot_warnings = True


def show_and_save(name, path=None, own_dir=True, extension=".pdf", tight_layout=True):
    from os.path import realpath, dirname, basename, splitext, join

    if not plot_path:
        raise ValueError("Call set_plot_path first!")
    if not path:
        this_file = realpath(plot_path)
        path = dirname(this_file)
        script, _ = splitext(basename(this_file))
        if own_dir:
            path = join(path, "plots", script)
        else:
            path = join(path, "plots")
        mkdir_p(path)
    filename = join(path, name + extension)

    if not disable_plot_warnings:
        warning = plt.figtext(0.5, 0.5, str(datetime.date.today()), color="gray")

    if tight_layout:
        plt.tight_layout()
    # figure = plt.gcf()
    # figure.set_size_inches(5,3)
    try:
        plt.savefig(filename)
        if not disable_plot_warnings:
            warning.remove()
        plt.show(block=False)
    except:
        print("Problems with plotting, trying again with different backend…")
        try:
            matplotlib.use("agg")
            plt.savefig(filename)
            if not disable_plot_warnings:
                warning.remove()
            plt.show(block=False)
            print("Plotting successful.")
        except:
            print("Plotting failed.")
    plt.clf()


plt.show_and_save = show_and_save
plt.set_plot_path = set_plot_path


class colormap_special_zero:
    # adapted from https://matplotlib.org/3.1.0/gallery/userdemo/colormap_normalizations_diverging.html#sphx-glr-gallery-userdemo-colormap-normalizations-diverging-py
    def __init__(self, values=None, limits=None, zero_point=0):
        normal_map = cm.viridis(np.linspace(0, 1, 256))
        zero_map = cm.Greys(np.linspace(0, 0.01, 256))
        all_colors = np.vstack((zero_map, normal_map))
        self.cmap = colors.LinearSegmentedColormap.from_list("special_zero", all_colors)

        self.zero_point = zero_point
        if values is not None:
            self.vmin = np.min(values)
            self.vmax = np.max(values)
        elif limits:
            self.vmin, self.vmax = limits
        else:
            raise ValueError("colormap_special_zero needs either `values` or `limits`")
        if self.vmin == 0:
            self.vmin = -self.vmax * 0.05
        if zero_point == 0:
            zero_point = min(0.01 * self.vmax, 0.1)

        if zero_point <= self.vmin:  # white colormap isn't needed
            self.norm = None
            self.cmap = cm.viridis
        else:
            self.norm = colors.DivergingNorm(
                vmin=self.vmin, vcenter=zero_point, vmax=self.vmax
            )

    def get_plot_config(self):
        return dict(norm=self.norm, cmap=self.cmap)


# import matplotlib
# https://stackoverflow.com/questions/11367736/matplotlib-consistent-font-using-latex
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'


class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


from box import Box

# TODO: Rename saved_values to lab_notebook
saved_values_file = config.root + "saved_values.yaml"
saved_values_yaml_root = None


def set_saved_value_yaml_root_by_filename(__file__):
    from os.path import realpath, dirname, basename, splitext, join

    global saved_values_yaml_root
    saved_values_yaml_root = splitext(basename(__file__))[0]
    # remove the 'n01_' part of a filename like 'n01_xxx'
    saved_values_yaml_root = re.sub("^\w\d+\w?_", "", saved_values_yaml_root)


def save_value(name, value, format=None):
    try:
        b = Box.from_yaml(filename=saved_values_file)
    except (FileNotFoundError, box.exceptions.BoxError):
        b = Box()
        print("Created new saved values file.")

    # numpy scalars don't play well with Box
    if isinstance(value, int):
        value = int(value)
    if isinstance(value, float):
        value = float(value)

    if format:
        value = ("{:" + format + "}").format(value)
    if saved_values_yaml_root:
        if saved_values_yaml_root in b.keys():
            b[saved_values_yaml_root][name] = value
        else:
            b[saved_values_yaml_root] = {name: value}
    else:
        b[name] = value
    b.to_yaml(filename=saved_values_file)
    if saved_values_yaml_root:
        print(f"{saved_values_yaml_root}: {name} : {value}")
    else:
        print(f": {name} : {value}")


def save_table(name, df: pd.DataFrame, **kw_args):
    from os.path import realpath, dirname, basename, splitext, join

    if not plot_path:
        raise ValueError("Call set_plot_path first!")
    this_file = realpath(plot_path)
    path = dirname(this_file)
    script, _ = splitext(basename(this_file))
    path = join(path, "plots", script)
    mkdir_p(path)
    filename = join(path, name + ".tex")

    config = dict(
        index=False,  # don't print index
        float_format="%.2f",
        column_format="S" * len(df.columns),  # use SI unix table
        escape=False,  # keep $ characters, etc.
        label=name.replace(" ", "_"),  # make sensible label
        header=["{" + str(c) + "}" for c in df.columns],
    )
    config.update(kw_args)
    assert len(df.columns) == len(config["column_format"]) or len(
        df.columns
    ) + 1 == len(config["column_format"]), "this will produce an error in latex!"

    # sourround strings with braces to satisfy SI unix
    df = df.copy()  # don't mess around with references that might be used elsewhere
    for i in range(df.values.shape[0]):
        for j in range(df.values.shape[1]):
            try:
                f = float(df.values[i, j])
            except ValueError:
                df.iloc[i, j] = "{" + df.values[i, j].replace("_", r"\_") + "}"

    try:
        table = df.to_latex(**config).split("\n")
    except TypeError: # happens when floats and str are mixed: workaround
        for i in range(df.values.shape[0]):
            for j in range(df.values.shape[1]):
                f = df.values[i, j]
                if isinstance(f,float) or isinstance(f,str):
                    try:
                        df.iloc[i, j] = format(f,config["float_format"][1:])
                    except:
                        pass
        table = df.to_latex(**config).split("\n")
        
    # remove table environment to do this later in Latex
    if "\\begin{table}" in table[0]:
        table = table[1:]
        if "\\end{table}" in table[-1]:
            table = table[:-1]
        elif "\\end{table}" in table[-2]:
            table = table[:-2]
    with open(filename, "w") as tf:
        tf.write("\n".join(table))
