import argparse
import os
import operator
import sys
from functools import reduce
import pdb
import glob

from dotmap import DotMap


def get_from_dict(dictionary, key_list):
    """Get value from dictionary with arbitrary depth via descending list of keys."""
    return reduce(operator.getitem, key_list, dictionary)


def set_in_dict(dictionary, key_list, value):
    """Set value from dictionary with arbitrary depth via descending list of keys."""
    get_from_dict(dictionary, key_list[:-1])[key_list[-1]] = value


def listdir_nohidden(path):
    return glob.glob(os.path.join(path, "*"))


def file_exists(prospective_file):
    """Check if the prospective file exists"""
    file_path = os.path.join(os.getcwd(), prospective_file)
    if not os.path.exists(file_path):
        raise argparse.ArgumentTypeError("File: '{0}' does not exist".format(file_path))
    return file_path


def dir_exists_write_privileges(prospective_dir):
    """Check if the prospective directory exists with write privileges."""
    dir_path = os.path.join(os.getcwd(), prospective_dir)
    if not os.path.isdir(dir_path):
        raise argparse.ArgumentTypeError(
            "Directory: '{0}' does not exist".format(dir_path)
        )
    elif not os.access(dir_path, os.W_OK):
        raise argparse.ArgumentTypeError(
            "Directory: '{0}' is not writable".format(dir_path)
        )
    return dir_path


def dir_exists_read_privileges(prospective_dir):
    """Check if the prospective directory exists with read privileges."""
    dir_path = os.path.join(os.getcwd(), prospective_dir)
    if not os.path.isdir(dir_path):
        raise argparse.ArgumentTypeError(
            "Directory: '{0}' does not exist".format(dir_path)
        )
    elif not os.access(dir_path, os.R_OK):
        raise argparse.ArgumentTypeError(
            "Directory: '{0}' is not readable".format(dir_path)
        )
    return dir_path


def override_dotmap(namespace: argparse.Namespace, config_key: str) -> DotMap:
    """Recursively override keys in a DotMap given CLI arguments """
    cfg = getattr(namespace, config_key)

    # Place all CLI arguments as top-level keys in config
    for arg in vars(namespace):
        if arg is not config_key and getattr(namespace, arg) is not None:
            cfg[arg] = getattr(namespace, arg)

    return cfg
