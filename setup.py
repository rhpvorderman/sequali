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

import os
import platform
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup


def extra_compile_args():
    if sys.platform.startswith("linux") and platform.machine() == "x86_64":
        # SEQUALI_CPU_BASIC=1 pip install --no-binary sequali sequali
        # Will work for linux users that do not have SSSE3 support.
        if not os.getenv("SEQUALI_CPU_BASIC"):
            return ["-mssse3"]
    # Do not bother with Windows and MacOS as it is not given that they have
    # a compiler installed. Simply use compatible wheels instead so no users
    # run into trouble.
    return None


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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: "
        "GNU Affero General Public License v3 or later (AGPLv3+)",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",  # SupportsIndex requires 3.8
    install_requires=[
        "xopen>=1.8.0",
        "pygal>=3.0.0",
        "tqdm"
    ],
    package_data={'sequali': ['*.c', '*.h', '*.pyi', 'py.typed',
                              'contaminants/*', 'adapters/*']},
    ext_modules=[
        Extension("sequali._qc", ["src/sequali/_qcmodule.c"],
                  extra_compile_args=extra_compile_args())],
    entry_points={"console_scripts": [
        'sequali=sequali.__main__:main',
        'sequali-report=sequali.__main__:sequali_report'
    ]},
)
