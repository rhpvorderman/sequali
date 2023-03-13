/*
Copyright (C) 2023 Leiden University Medical Center
This file is part of fasterqc

fasterqc is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

fasterqc is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with fasterqc.  If not, see <https://www.gnu.org/licenses/
*/

#define PY_SSIZE_T_CLEAN
#include "Python.h"

PyTypeObject *SequenceRecord;

/* Nice trick from fastp: A,C, G, T, N all have different last three
   bits. So this requires only 8 entries per count array. Fastp performs
   a bitwise and of 0b111 on every character.
   This can be taken further by using  a lookup table. A=1, C=2, G=3, T=4.
   Lowercase a,c,g and t are supported. All other characters are index 0 and
   are considered N. This way we can make a very dense count table, and don't
   have to check every nucleotide if it is within bounds. Furthermore, odd
   characters such as IUPAC K will map to N, unlike the fastp method where K
   will map to C. */

const uint8_t NUCLEOTIDE_TO_INDEX[128] = {
// Control characters
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// Interpunction numbers etc
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//  A, B, C, D, E, F, G, H, I, J, K, L, M, N, O,
    0, 1, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0,
//  P, Q, R, S, T, U, V, W, X, Y, Z,  
    0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//  a, b, c, d, e, f, g, h, i, j, k, l, m, n, o,
    0, 1, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0,
//  p, q, r, s, t, u, v, w, x, y, z, 
    0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
};

#define NUC_TABLE_SIZE 5
#define PHRED_MAX 93 
#define PHRED_LIMIT 47
#define PHRED_TABLE_SIZE ((PHRED_LIMIT / 4) + 1)

typedef uint64_t counter_t;

/* Illumina reads often use a limited set of phreds rather than the full range
   of 0-93. Putting phreds before nucleotides in the array type therefore gives
   nice dense spots in the array where all nucleotides with the same phred sit
   next to eachother. That leads to better cache locality. */
typedef counter_t counttable_t[PHRED_TABLE_SIZE][NUC_TABLE_SIZE];

inline uint8_t phred_to_index(uint8_t phred) {
    if (phred > PHRED_LIMIT){
        phred = PHRED_LIMIT;
    }
    return phred >> 2;
}

typedef struct _QCMetricsStruct {
    PyObject_HEAD
    PyObject *seq_name;
    PyObject *qual_name;
    uint8_t phred_offset;
    counttable_t *count_tables;
    Py_ssize_t max_length;
    size_t number_of_reads;
} QCMetrics;

static void
QCMetrics_dealloc(QCMetrics *self) {
    Py_DECREF(self->seq_name);
    Py_DECREF(self->qual_name);
    PyMem_Free(self->count_tables);
}

static PyObject *
QCMetrics__new__(PyTypeObject *type, PyObject *args, PyObject *kwargs){
    static char *kwargnames[] = {NULL};
    static char *format = ":QCMetrics";
    uint8_t phred_offset = 33;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, kwargnames)) {
        return NULL;
    }
    PyObject *seq_name = PyUnicode_FromString("sequence");
    if (seq_name == NULL) {
        return NULL;
    }
    PyObject *qual_name = PyUnicode_FromString("qualities");
    if (qual_name == NULL) {
        return NULL;
    }

    counttable_t *count_tables = NULL;
    QCMetrics *self = PyObject_New(QCMetrics, type);
    self->max_length = 0;
    self->phred_offset = phred_offset;
    self->count_tables = NULL;
    self->number_of_reads = 0;
    self->seq_name = seq_name; 
    self->qual_name = qual_name;
}

static int
QCMetrics_resize(QCMetrics *self, Py_ssize_t new_size) 
{
    counttable_t *tmp = PyMem_Realloc(
        self->count_tables, new_size * sizeof(counttable_t)
    );
    if (tmp == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    self->count_tables = tmp;
    /* Set the added part to 0 */
    memset(self->count_tables + self->max_length, 0, 
           (new_size - self->max_length) * sizeof(counttable_t));
    self->max_length = new_size;
}

static PyObject * 
QCMetrics_add_read(QCMetrics *self, PyObject *read) 
{
    if (Py_TYPE(read) != SequenceRecord) {
        PyErr_Format(PyExc_TypeError, 
                     "Read should be a dnaio.SequenceRecord object, got %s", 
                     Py_TYPE(read)->tp_name);
        return NULL;
    }
    /* PyObject_GetAttrString creates a new unicode object every single time. 
       Use PyObject_GetAttr to prevent this. */
    PyObject *sequence_obj = PyObject_GetAttr(read, self->seq_name);
    PyObject *qualities_obj = PyObject_GetAttr(read, self->qual_name);
    /* Dnaio guarantees ASCII strings */
    const uint8_t *sequence = PyUnicode_DATA(sequence_obj);
    const uint8_t *qualities = PyUnicode_DATA(qualities_obj);
    /* Dnaio guarantees same length */
    Py_ssize_t sequence_length = PyUnicode_GET_LENGTH(sequence_obj);
    size_t i;
    uint8_t phred_offset = self->phred_offset;
    uint8_t c, q, c_index, q_index;

    if (sequence_length > self->max_length) {
        QCMetrics_resize(self, sequence_length);
    }

    self->number_of_reads += 1; 
    for (i=0; i < (size_t)sequence_length; i+=1) {
        c = sequence[i];
        q = qualities[i] - phred_offset;
        if (q > PHRED_MAX) {
            PyErr_Format(
                PyExc_ValueError, 
                "Not a valid phred character: %c", qualities[i]
            );
            return NULL;
        }
        q_index = phred_to_index(q);
        c_index = NUCLEOTIDE_TO_INDEX[c];
        self->count_tables[i][q_index][c_index] += 1;
    }
    Py_RETURN_NONE;
}

QCMetrics_count_table_view(QCMetrics *self, PyObject *Py_UNUSED(ignore))
{
    return PyMemoryView_FromMemory(
        (char *)self->count_tables,
        self->max_length * sizeof(counttable_t),
        PyBUF_READ
    );
}

static struct PyModuleDef _qc_module = {
    PyModuleDef_HEAD_INIT,
    "_qc",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,
    NULL,  /* module methods */
};

PyMODINIT_FUNC
PyInit__qc(void)
{
    PyObject *m = PyModule_Create(&_qc_module);
    if (m == NULL) {
        return NULL;
    }
    PyObject *dnaio = PyImport_ImportModule("dnaio");
    if (dnaio == NULL) {
        return NULL;
    }
    PyTypeObject *SequenceRecord = 
        (PyTypeObject *)PyObject_GetAttrString(dnaio, "SequenceRecord");
    if (SequenceRecord == NULL) {
        return NULL;
    }
    if (!PyType_CheckExact(SequenceRecord)) {
        PyErr_Format(PyExc_RuntimeError, 
            "SequenceRecord is not a type class but, %s", 
            Py_TYPE(SequenceRecord)->tp_name);
        return NULL;
    }
    return m;
}