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

struct Entry {
    Py_ssize_t score;
    Py_ssize_t query_matches;
};

PyDoc_STRVAR(sequence_identity__doc__, 
"Calculate sequence identity based on a smith-waterman matrix. Only keep\n"
"two columns in memory as no walk-back is needed.\n"
"Identity is given as (query_length - errors / query_length).\n"
);

#define sequence_identity_method METH_VARARGS | METH_KEYWORDS

static PyObject * 
sequence_identity(PyObject *module, PyObject *args, PyObject *kwargs)
{
    static char *format = "UU|nnnn:identify_sequence";
    static char *kwnames[] = {
        "target", "query", "match_score", "mismatch_penalty", 
        "deletion_penalty", "inertion_penalty", NULL
    };
    PyObject *target_obj = NULL; 
    PyObject *query_obj = NULL;
    Py_ssize_t match_score = 1; 
    Py_ssize_t mismatch_penalty = -1; 
    Py_ssize_t deletion_penalty = -1; 
    Py_ssize_t insertion_penalty = -1;
    if (!PyArg_ParseTupleAndKeywords(
        args, kwargs, format, kwnames, 
        &target_obj, &query_obj, &match_score, &mismatch_penalty, 
        &deletion_penalty, &insertion_penalty)
    ) {
        return NULL;
    }
    if (!PyUnicode_IS_COMPACT_ASCII(target_obj)) {
        PyErr_Format(
            PyExc_ValueError,
            "Only ascii strings are allowed. Got %R",
            target_obj
        );
        return NULL;
    }
    if (!PyUnicode_IS_COMPACT_ASCII(query_obj)) {
        PyErr_Format(
            PyExc_ValueError,
            "Only ascii strings are allowed. Got %R",
            target_obj
        );
        return NULL;
    }
    const uint8_t *target = PyUnicode_DATA(target_obj);
    const uint8_t *query = PyUnicode_DATA(query_obj);
    Py_ssize_t target_length = PyUnicode_GET_LENGTH(target_obj);
    Py_ssize_t query_length = PyUnicode_GET_LENGTH(query_obj);
    if (query_length > 31) {
        PyErr_Format(
            PyExc_ValueError,
            "Only query with lengths less than 32 are supported. Got %zd",
            query_length
        );
        return NULL;
    }
    Py_ssize_t highest_score = 0;
    Py_ssize_t most_matches = 0;
    struct Entry prev_column[32];
    struct Entry new_column[32];
    memset(prev_column, 0, 32 * sizeof(struct Entry));
    memset(new_column, 0, 32 * sizeof(struct Entry));
    for (Py_ssize_t i=0; i < target_length; i++) {
        for (Py_ssize_t j=1; j < query_length + 1; j++) {
            uint8_t target_char = target[i];
            uint8_t query_char = query[j - 1];
            struct Entry prev_entry = prev_column[j-1];
            Py_ssize_t linear_score; 
            Py_ssize_t linear_matches;
            if (target_char == query_char) {
                linear_score = prev_entry.score + match_score;
                linear_matches = prev_entry.query_matches + 1;
            } else {
                linear_score = prev_entry.score + mismatch_penalty;
                linear_matches = prev_entry.query_matches;
            }
            struct Entry prev_ins_entry = prev_column[j];
            struct Entry prev_del_entry = new_column[j - 1];
            Py_ssize_t insertion_score = prev_ins_entry.score + insertion_penalty;
            Py_ssize_t deletion_score = prev_del_entry.score + deletion_penalty;
            Py_ssize_t score;
            Py_ssize_t matches; 
            if (linear_score >= insertion_score && linear_score >= deletion_score) {
                score = linear_score;
                matches = linear_matches;
            } else if (insertion_score >= deletion_score) {
                /* When an insertion happens in the query in theory we can 
                   match all query characeters still. So deduct one as a penalty. */
                score = insertion_score;
                matches = prev_ins_entry.query_matches - 1;
            } else {
                /* When a deletion happens in the query, that character cannot
                   match anything anymore. So no need to deduct a penalty. */
                score = deletion_score;
                matches = prev_del_entry.query_matches;
            }
            if (score < 0) {
                score = 0;
                matches = 0;
            }
            new_column[j].score = score;
            new_column[j].query_matches = matches;
            if (score == highest_score && matches > most_matches) {
                most_matches = matches;
            } else if (score > highest_score) {
                highest_score = score;
                most_matches = matches;
            }
        }
        memcpy(prev_column, new_column, sizeof(prev_column));
    }
    double identity = (double)most_matches / (double)query_length;
    return PyFloat_FromDouble(identity);
}

static PyMethodDef _seqident_methods[] = {
    {"sequence_identity", (PyCFunction)sequence_identity, 
     sequence_identity_method, sequence_identity__doc__},
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
