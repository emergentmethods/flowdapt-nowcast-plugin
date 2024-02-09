from importlib import resources
from pathlib import Path


def get_package_datafile_path(filepath: str, package_name: str) -> Path:
    """
    Get the path of a file in the package data directory.

    :param filepath: Path to file in package data directory.
    :type filepath: str
    :param package_name: Name of package.
    :type package_name: str
    :return: The absolute file path.
    :rtype: str
    """
    return resources.files(package_name) / filepath
