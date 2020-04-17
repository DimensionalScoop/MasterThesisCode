from datetime import datetime
import matplotlib.pyplot as plt
import os
import pickle

status_report_path = ""


def init(model=None, hint="", add_info=None):
    global status_report_path
    path = "reports/" + str(datetime.now().strftime("%m%d-%H%M%S"))
    if hint != "":
        path += " " + hint
    path += "/"
    if not os.path.exists(path):
        os.makedirs(path)
    status_report_path = path

    if model is not None:
        save_model(model)

    if add_info is not None:
        save_README(add_info)


def unique_filename(filename):
    if not os.path.exists(status_report_path + filename):
        return filename
    nr = 0
    filename, extension = filename.split(".")
    while os.path.exists(status_report_path + filename + "-%d." % nr + extension):
        nr += 1
    return status_report_path + filename + "-%d." % nr + extension


def save_README(add_info):
    with open(status_report_path + "README", "a") as f:
        try:
            f.writelines(add_info)
        except:
            f.write(add_info)


def save_model(model, name_hint=""):
    global status_report_path
    assert status_report_path != ""

    filename = unique_filename(status_report_path + name_hint + "model.hdf")
    model.save(filename)


def save_plot(name_hint=""):
    svg_path = status_report_path + "svg/"
    if not os.path.exists(svg_path):
        os.makedirs(svg_path)

    filename_pdf = unique_filename(status_report_path + name_hint + ".pdf")
    filename_svg = unique_filename(svg_path + name_hint + ".svg")

    plt.savefig(filename_pdf)
    plt.savefig(filename_svg)

    # dump data contained in plot
    plot_data = []
    for line in plt.gca().get_lines():
        la = line.get_label()
        xd = line.get_xdata()
        yd = line.get_ydata()
        plot_data.append((la, xd, yd))

    if not os.path.exists(status_report_path + "plot_data/"):
        os.makedirs(status_report_path + "plot_data/")
    save_obj(plot_data, "plot_data/" + name_hint)


def save_obj(obj, name_hint=""):
    filename = unique_filename(status_report_path + name_hint + ".pickle")
    pickle.dump(obj, open(filename, "bw"))
