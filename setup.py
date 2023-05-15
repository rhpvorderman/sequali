# Copyright (C) 2023 Leiden University Medical Center
# This file is part of sequali
#
# sequali is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# sequali is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with sequali.  If not, see <https://www.gnu.org/licenses/

from pathlib import Path

from setuptools import Extension, find_packages, setup

setup(
    name="sequali",
    version="0.1.0-dev",
    description="Fast FASTQ quality metrics",
    author="Leiden University Medical Center",
    author_email="r.h.p.vorderman@lumc.nl",
    long_description=Path("README.rst").read_text(),
    long_description_content_type="text/x-rst",
    license="AGPL-3.0-or-later",
    keywords="FASTQ QC",
    zip_safe=False,
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url="https://github.com/rhpvorderman/sequali",
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: "
        "GNU Affero General Public License v3 or later (AGPLv3+)",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.7",
    install_requires=[
        "xopen",
        "pygal>=3.0.0"
    ],
    package_data={'sequali': ['*.c', '*.h', '*.pyi', 'py.typed']},
    ext_modules=[Extension("sequali._qc", ["src/sequali/_qcmodule.c"])],
    entry_points={"console_scripts": ['sequali=sequali:main']},
)
