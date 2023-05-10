from setuptools import find_packages, setup
from typing import List


E_DOT = '-e .'


# define a function to read the requirements from the requirements.txt file
def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if E_DOT in requirements:
            requirements.remove(E_DOT)

    

    return requirements


setup(
    name="ML Project-1 :-  Student Performance analysis",
    version="0.0.1",
    author="Yrana",
    author_email="yrana7829@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
