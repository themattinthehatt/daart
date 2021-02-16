"""Utility functions for daart package."""

import os


def make_dir_if_not_exists(save_file):
    """Utility function for creating necessary dictories for a specified filename.

    Parameters
    ----------
    save_file : str
        absolute path of save file

    """
    save_dir = os.path.dirname(save_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
