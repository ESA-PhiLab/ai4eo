import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ai4eo",
    version="0.0.1",
    author="John Mrziglod",
    author_email="john.mrziglod@esa.int",
    description="Python routines of Artificial Intelligence applications for Earth Observation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ESA-PhiLab/AI4EO",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

