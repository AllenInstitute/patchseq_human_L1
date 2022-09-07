from setuptools import setup, find_packages
import glob

# with open('requirements.txt', 'r') as f:
#     required = f.read().splitlines()
required = [
    'numpy',
    'pandas',
    'seaborn',
    'matplotlib',
    'statsmodels',
    'allensdk'
]

setup(
    name = 'patchseq_utils',
    version = '0.1.0',
    description = """Tools supporting patch-seq data analysis""",
    author = "Thomas Chartrand",
    author_email = "tom.chartrand@alleninstitute.org",
    url = '',
    packages = find_packages(),
    install_requires = required,
    python_requires=">=3.6",
    include_package_data=True,
)
