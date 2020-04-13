from argparse import ArgumentParser

from utils.file import load_training_config_file
from utils.argparse_util import *


def add_common_arguments(parser: ArgumentParser) -> ArgumentParser:

    core = parser.add_argument_group("core arguments")

    core.add_argument(
        "--config-file",
        type=load_training_config_file,
        default="config/default.yaml",
        help="Config file absolute path. CLI takes priority over config file",
    )

    core.add_argument(
        "--log-port", type=int, optional=True, help="Port number of logging server"
    )

    core.add_argument(
        "--gpu-id", type=int, optional=True, help="GPU id for single gpu training"
    )

    core.add_argument(
        "--log-dir",
        type=dir_exists_write_privileges,
        optional=True,
        help="Log file storage directory path",
    )

    core.add_argument(
        "--data-path",
        type=dir_exists_read_privileges,
        optional=True
        help="Data file storage directory path",
    )

    core.add_argument(
        "--restore-weights",
        type=file_exists,
        help="Restore model with pre-trained weights",
    )

    return parser
