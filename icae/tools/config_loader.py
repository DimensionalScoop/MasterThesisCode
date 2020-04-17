import yaml
import errno
import os
import sys

python2 = sys.version_info.major < 3
if python2:
    FileNotFoundError = IOError


def mkdir_p(path):
    """python2 way of running 'mkdir -p [path]'"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class Config:
    def __init__(self, dicts, root_dir):
        for i in dicts:
            if isinstance(dicts[i], dict):
                self.__dict__.update({i: Config(dicts[i], root_dir)})
            else:
                self.__dict__.update({i: dicts[i]})

            self.root = root_dir


path = os.path.dirname(os.path.abspath(__file__)) + "/"
for config_file in ["config.yml", "../config.yml", "../../config.yml"]:
    try:  # search for the correct path
        dicts = yaml.safe_load(open(path + config_file))
        # all paths in the config file a relative to this root
        root_dir = os.path.dirname(os.path.realpath(path + config_file)) + "/"
        print("Root dir: `%s`."%root_dir)
        break
    except FileNotFoundError as exc:
        pass
if dicts is None:
    raise FileNotFoundError("Where is the config.yml file?")

config = Config(dicts, root_dir)
