from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """
    Reads a requirements.txt file and returns a list of requirements,
    excluding the editable install flag "-e ."
    """
    with open(file_path) as file_obj:
        requirements = file_obj.read().splitlines()
        requirements = [req.strip() for req in requirements if req.strip() and req.strip() != HYPHEN_E_DOT]
    return requirements

setup(
    name="HotelBookingsProject",
    version="0.0.1",
    author="Omido",
    description="An end-to-end machine learning project to predict hotel booking cancellation.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=get_requirements("requirements.txt")
)
