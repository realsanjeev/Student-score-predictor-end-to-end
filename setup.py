from setuptools import find_packages, setup
from pathlib import Path

E_DASH_DOT = "-e ."
README_PATH = "README.md"
REQUIREMENTS_FILE_PATH = "requirements.txt"

def get_requirements(file_path: Path) -> list():
    """
    Get package name from `requirements.txt`

    Args:
        file_path: Path -> file_path for requirements
    Results:
        requirements: List -> list of name of package
    """
    try:
        with open(file_path) as file_handler:
            requirements = file_handler.readlines()
    except FileNotFoundError:
        raise (f"`requirements.txt` doesnot exist in path. \
               Place `requirement.txt` file in same folder along with `setup.py`")
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
        decription: str -> Description of project
    """
    try:
        with open(file_path) as file_handler:
            content = file_handler.read()
    except FileExistsError:
        print(f"[ERROR] No README.md file found for long description")
        content = ''
    return content


setup(
    name="ml project",
    version="0.1",
    description="Setup file for ML project",
    author="realsanjeev",
    author_email="realsanjeev2@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements(REQUIREMENTS_FILE_PATH),
    long_description=get_readme(README_PATH),
)