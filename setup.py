from setuptools import Command, find_packages, setup

__lib_name__ = "FlatST"
__lib_version__ = "1.0.1"
__description__ = "FlatST: An efficient and stable domain detection framework"
__url__ = "https://github.com/CHENszu?tab=repositories"
__author__ = "Xudong Chen"
__author_email__ = "chxd6266@gmail.com"
__license__ = "MIT"
__keywords__ = ["domain detection",  "Smoothing mechanism"]
__requires__ = ["requests",]

with open("README.rst", "r", encoding="utf-8") as f:
    __long_description__ = f.read()

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    packages = ['FlatST'],
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True,
    long_description = __long_description__
)
