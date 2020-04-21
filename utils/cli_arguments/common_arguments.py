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

    core.add_argument("--log-port", type=int, help="Port number of logging server")

    core.add_argument("--gpu-id", type=int, help="GPU id for single gpu training")

    core.add_argument(
        "--trajectory-length", type=int, help="Trajectory length to use for training"
    )

    core.add_argument(
        "--log-dir",
        type=dir_exists_write_privileges,
        help="Log file storage directory path",
    )

    core.add_argument(
        "--forward-save-path",
        type=dir_exists_write_privileges,
        help="Forward model storage directory path",
    )

    core.add_argument(
        "--gan-save-path",
        type=dir_exists_write_privileges,
        help="GAN model storage directory path",
    )

    core.add_argument(
        "--train-data-path",
        type=dir_exists_read_privileges,
        help="Train data file storage directory path",
    )

    core.add_argument(
        "--evaluation-data-path",
        type=dir_exists_read_privileges,
        help="Evaluation data file storage directory path",
    )

    core.add_argument(
        "--restore-weights",
        type=file_exists,
        help="Restore model with pre-trained weights",
    )

    return parser
