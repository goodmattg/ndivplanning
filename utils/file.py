import logging
import sys
import os
import yaml
import imp
import pprint
import argparse
import pickle as pkl
import pdb

from dotmap import DotMap

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.DEBUG,
    stream=sys.stdout,
)


def load_training_config_file(filename):
    """Load a training configuration yaml file into a DotMap dictionary."""
    print("Loading training configuration file: {0}".format(filename))
    config_file_path = os.path.join(os.getcwd(), filename)

    with open(config_file_path, "r") as stream:
        cfg = DotMap(yaml.load(stream))

    cfg = make_paths_absolute(os.getcwd(), cfg)

    return cfg


def make_paths_absolute(dir_, cfg: DotMap) -> DotMap:
    """
    Make all values for keys ending with `_path` absolute to dir_.

    Parameters
    ----------
    dir_ : str
    cfg : dict

    Returns
    -------
    cfg : dict
    """
    for key in cfg.keys():
        if key.endswith("_path"):
            cfg[key] = os.path.join(dir_, cfg[key])
            cfg[key] = os.path.abspath(cfg[key])
            if not os.path.isfile(cfg[key]):
                logging.error("%s does not exist.", cfg[key])
        if type(cfg[key]) is DotMap:
            cfg[key] = make_paths_absolute(dir_, cfg[key])
    return cfg
