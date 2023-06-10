from setuptools import find_packages, setup

E_DASH_DOT = "-e ."

def get_requirements(filename: str) -> list():
    """
    Get package name from `requirements.txt`

    Args:
        filename: str -> filename for requirements
    Results:
        list of name of package
    """
    with open(filename) as file_handler:
       requirements = file_handler.readlines()
    requirements = [req.replace("\n", "") for req in requirements]
    if E_DASH_DOT in requirements:
        requirements.remove(E_DASH_DOT)
    return requirements

setup(
    name="ml project",
    version="0.1",
    description="Setup file for ml project",
    author="realsanjeev",
    author_email="realsanjeev2@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)