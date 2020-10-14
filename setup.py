import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EEG-Artifact-Removal", # Replace with your own username
    version="0.0.1",
    author="Cristhoper Ochoa Gutierrez",
    author_email="craisochoa22@gmail.com",
    description="A custom package for my thesis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/curais/EEG-Artifact-Removal",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)