# standard setup file to describe the package
from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = "-e ."


def get_requirements(filename: str) -> List[str]:
    """
    This method will read the requirements.txt file and return the list of packages.
    We will remove the HYPHEN_E_DOT from the list of packages.
    """
    requirements = []
    with open(filename) as f:
        for requirement in f:
            requirements.append(requirement.strip().replace("\n", ""))

    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    return requirements


setup(
    name="mlproject",
    version="0.0.1",
    author="Debu Sinha",
    author_email="debusinha2009@gmail.com",
    packages=find_packages(where=".", exclude=("tests",)),
    install_requires=get_requirements("requirements.txt"),
)
