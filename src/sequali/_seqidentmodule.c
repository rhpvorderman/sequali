/*
Copyright (C) 2023 Leiden University Medical Center
This file is part of Sequali

Sequali is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

Sequali is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with Sequali.  If not, see <https://www.gnu.org/licenses/
*/

#include "Python.h"

static PyObject * 
identify_sequence(PyObject *module, PyObject *args, PyObject *kwargs);

static PyMethodDef _seqident_methods[] = {
    {NULL},
};

static struct PyModuleDef _seqident_module = {
    PyModuleDef_HEAD_INIT,
    "_seqident",
    NULL, /* Module documentation*/
    -1, 
    _seqident_methods,
    .m_slots = NULL,
};

PyMODINIT_FUNC
PyInit__seqident(void)
{
    PyObject *m = PyModule_Create(&_seqident_module);
    if (m == NULL) {
        return NULL;
    }
    return m;
}
