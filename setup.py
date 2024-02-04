from typing import List
from setuptools import find_packages, setup
from pathlib import Path

E_DASH_DOT = "-e ."
README_PATH = Path("README.md")
REQUIREMENTS_FILE_PATH = Path("requirements.txt")

def get_requirements(file_path: Path) -> List[str]:
    """
    Get package name from `requirements.txt`

    Args:
        file_path: Path -> file_path for requirements
    Results:
        requirements: List[str] -> list of name of package
    """
    try:
        with open(file_path) as file_handler:
            requirements = file_handler.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"`requirements.txt` does not exist in path. \
               Place `requirements.txt` file in same folder along with `setup.py`")
    requirements = [req.replace("\n", "") for req in requirements]
    if E_DASH_DOT in requirements:
        requirements.remove(E_DASH_DOT)
    return requirements

def get_readme(file_path: Path) -> str:
    """
    Get description of project from `README.md` file if available

    Args:
        file_path: Path -> file_path for requirements
    Results:
        description: str -> Description of project
    """
    try:
        with open(file_path) as file_handler:
            content = file_handler.read()
    except FileNotFoundError:
        print(f"[ERROR] No README.md file found for long description")
        content = ''
    return content


setup(
    name="ml_project",
    version="0.1",
    description="Setup file for ML project",
    author="realsanjeev",
    author_email="realsanjeev2@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements(REQUIREMENTS_FILE_PATH),
    long_description=get_readme(README_PATH),
)