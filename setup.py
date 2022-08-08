from setuptools import setup, find_packages
import glob

# with open('requirements.txt', 'r') as f:
#     required = f.read().splitlines()

setup(
    name = 'patchseq_utils',
    version = '0.1.0',
    description = """Tools supporting patch-seq data analysis""",
    author = "Thomas Chartrand",
    author_email = "tom.chartrand@alleninstitute.org",
    url = '',
    packages = find_packages(),
    # install_requires = required,
    include_package_data=True,
)
