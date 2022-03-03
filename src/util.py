"""
This file contains utility functions and variables for the project
"""
import logging
from logging.handlers import WatchedFileHandler
from pathlib import Path
import os
import configparser


CONSTANT_NINF = float("-inf")


def get_dir_path(dir_name: str) -> str:
    """
    Get the directory path

    Args:
        dir_name: Directory name

    Returns:
        Directory path
    """
    curr_file_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(curr_file_dir)
    dir_path = os.path.join(parent_dir, dir_name)
    return dir_path


def read_config(config_name: str) -> configparser.ConfigParser:
    """
    Reads specified config file and return a ConfigParser object
    Please put all config files in 'configs' directory

    Args:
        config_name: Config file name

    Returns:
        ConfigParser object
    """
    config_dir_path = get_dir_path("configs")
    config_path = os.path.join(config_dir_path, config_name)
    parser = configparser.ConfigParser()
    parser.read(config_path)
    return parser


def init_logger(name: str) -> logging.Logger:
    """
    Initializes logger object

    Args:
        name: Logger name

    Returns:
        Global Logger object
    """
    log_dir = get_dir_path("log")
    log_file_path = os.path.join(log_dir, "eos.log")
    log_file = Path(log_file_path)
    log_file.touch(exist_ok=True)
    handler = WatchedFileHandler(log_file)
    formatter = logging.Formatter(logging.BASIC_FORMAT)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel("INFO")
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)

    return logger


def create_dirs() -> None:
    """
    Create the necessary directories if they do not exist

    Returns:
        None
    """
    dir_names = ["log", "models"]
    for dir_name in dir_names:
        dir_path = get_dir_path(dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
