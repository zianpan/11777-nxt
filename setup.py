from setuptools import setup, find_packages

setup(
    name="MMObject",  # Name of your package
    version="0.1",  # Version number
    packages=find_packages(),  # Automatically find all packages
    install_requires=[],  # Add dependencies here (if any)
    include_package_data=True,  # Include non-Python files specified in MANIFEST.in
)