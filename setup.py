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

# ABI3 building example from
# https://github.com/joerick/python-abi3-package-sample/blob/main/setup.py

from setuptools import Extension, setup

from wheel.bdist_wheel import bdist_wheel


class bdist_wheel_abi3(bdist_wheel):
    def get_tag(self):
        python, abi, plat = super().get_tag()

        if python.startswith("cp"):
            return "cp310", "abi3", plat
        return python, abi, plat


setup(
    ext_modules=[
        Extension(
            "sequali._qc",
            ["src/sequali/_qcmodule.c"],
            py_limited_api=True
        ),
        Extension(
            "sequali._seqident",
            ["src/sequali/_seqidentmodule.c"],
            py_limited_api=True),
    ],
    cmdclass={"bdist_wheel": bdist_wheel_abi3},
)
