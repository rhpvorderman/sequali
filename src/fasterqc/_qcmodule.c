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
#include "structmember.h"

#include "math.h"
#include "score_to_error_rate.h"

static PyTypeObject *SequenceRecord;

/* Nice trick from fastp: A,C, G, T, N all have different last three
   bits. So this requires only 8 entries per count array. Fastp performs
   a bitwise and of 0b111 on every character.
   This can be taken further by using  a lookup table. A=1, C=2, G=3, T=4.
   Lowercase a,c,g and t are supported. All other characters are index 0 and
   are considered N. This way we can make a very dense count table, and don't
   have to check every nucleotide if it is within bounds. Furthermore, odd
   characters such as IUPAC K will map to N, unlike the fastp method where K
   will map to C. */

static const uint8_t NUCLEOTIDE_TO_INDEX[128] = {
// Control characters
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// Interpunction numbers etc
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//     A, B, C, D, E, F, G, H, I, J, K, L, M, N, O,
    0, 1, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0,
//  P, Q, R, S, T, U, V, W, X, Y, Z,  
    0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//     a, b, c, d, e, f, g, h, i, j, k, l, m, n, o,
    0, 1, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0,
//  p, q, r, s, t, u, v, w, x, y, z, 
    0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
};
#define N 0
#define A 1
#define C 2
#define G 3
#define T 4
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

static inline uint8_t phred_to_index(uint8_t phred) {
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
    counter_t gc_content[101];
    counter_t phred_scores[PHRED_MAX + 1];
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

    QCMetrics *self = PyObject_New(QCMetrics, type);
    self->max_length = 0;
    self->phred_offset = phred_offset;
    self->count_tables = NULL;
    self->number_of_reads = 0;
    self->seq_name = seq_name; 
    self->qual_name = qual_name;
    memset(self->gc_content, 0, 101 * sizeof(counter_t));
    memset(self->phred_scores, 0, (PHRED_MAX + 1) * sizeof(counter_t));
    return (PyObject *)self;
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
    return 0;
}


PyDoc_STRVAR(QCMetrics_add_read__doc__,
"add_read($self, read, /)\n"
"--\n"
"\n"
"Add a read to the count metrics. \n"
"\n"
"  read\n"
"    A dnaio.SequenceRecord object.\n"
);

#define QCMETRICS_ADD_READ_METHODDEF    \
    {"add_read", (PyCFunction)(void(*)(void))QCMetrics_add_read, METH_O, \
     QCMetrics_add_read__doc__}

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
    counter_t base_counts[NUC_TABLE_SIZE] = {0, 0, 0, 0, 0};
    double accumulated_error_rate = 0.0;

    if (sequence_length > self->max_length) {
        if (QCMetrics_resize(self, sequence_length) != 0) {
            goto error;
        }
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
            goto error;
        }
        q_index = phred_to_index(q);
        c_index = NUCLEOTIDE_TO_INDEX[c];
        self->count_tables[i][q_index][c_index] += 1;
        base_counts[c_index] += 1;
        accumulated_error_rate += SCORE_TO_ERROR_RATE[q];
    }
    counter_t at_counts = base_counts[A] + base_counts[T];
    counter_t gc_counts = base_counts[C] + base_counts[G];
    double gc_content_percentage = (double)at_counts * (double)100.0 / (double)(at_counts + gc_counts);
    uint64_t gc_content_index = (uint64_t)round(gc_content_percentage);
    assert(gc_content_index >= 0);
    assert(gc_content_index <= 100);
    self->gc_content[gc_content_index] += 1;

    double average_error_rate = accumulated_error_rate / (double)sequence_length;
    double average_phred = -10.0 * log10(average_error_rate);
    uint64_t phred_score_index = (uint64_t)round(average_phred);
    assert(phred_score_index >= 0);
    assert(phred_score_index <= PHRED_MAX);
    self->phred_scores[phred_score_index] += 1;

    Py_DECREF(sequence_obj);
    Py_DECREF(qualities_obj);
    Py_RETURN_NONE;
error:
    Py_DECREF(sequence_obj);
    Py_DECREF(qualities_obj);
    return NULL;
}

PyDoc_STRVAR(QCMetrics_count_table_view__doc__,
"count_table_view($self, /)\n"
"--\n"
"\n"
"Return a memoryview on the produced count table. \n"
);

#define QCMETRICS_COUNT_TABLE_VIEW_METHODDEF    \
    {"count_table_view", (PyCFunction)(void(*)(void))QCMetrics_count_table_view, \
     METH_NOARGS, QCMetrics_count_table_view__doc__}

static PyObject *
QCMetrics_count_table_view(QCMetrics *self, PyObject *Py_UNUSED(ignore))
{
    return PyMemoryView_FromMemory(
        (char *)self->count_tables,
        self->max_length * sizeof(counttable_t),
        PyBUF_READ
    );
}

PyDoc_STRVAR(QCMetrics_gc_content_view__doc__,
"gc_content_view($self, /)\n"
"--\n"
"\n"
"Return a memoryview on the produced gc content counts. \n"
);

#define QCMETRICS_GC_CONTENT_VIEW_METHODDEF    \
    {"gc_content_view", (PyCFunction)(void(*)(void))QCMetrics_gc_content_view, \
     METH_NOARGS, QCMetrics_gc_content_view__doc__}

static PyObject *
QCMetrics_gc_content_view(QCMetrics *self, PyObject *Py_UNUSED(ignore))
{
    return PyMemoryView_FromMemory(
        (char *)self->gc_content,
        101 * sizeof(counter_t),
        PyBUF_READ
    );
}

PyDoc_STRVAR(QCMetrics_phred_scores_view__doc__,
"phred_scores_view($self, /)\n"
"--\n"
"\n"
"Return a memoryview on the produced average phred score counts. \n"
);

#define QCMETRICS_PHRED_SCORES_VIEW_METHODDEF    \
    {"phred_scores_view", (PyCFunction)(void(*)(void))QCMetrics_phred_scores_view, \
     METH_NOARGS, QCMetrics_phred_scores_view__doc__}

static PyObject *
QCMetrics_phred_scores_view(QCMetrics *self, PyObject *Py_UNUSED(ignore))
{
    return PyMemoryView_FromMemory(
        (char *)self->phred_scores,
        (PHRED_MAX + 1) * sizeof(counter_t),
        PyBUF_READ
    );
}

PyMethodDef QCMetrics_methods[] = {
    QCMETRICS_ADD_READ_METHODDEF,
    QCMETRICS_COUNT_TABLE_VIEW_METHODDEF,
    QCMETRICS_GC_CONTENT_VIEW_METHODDEF,
    QCMETRICS_PHRED_SCORES_VIEW_METHODDEF,
    {NULL},
};

PyMemberDef QCMetrics_members[] = {
    {"max_length", T_PYSSIZET, offsetof(QCMetrics, max_length), READONLY, 
     "The length of the longest read"},
    {"number_of_reads", T_ULONGLONG, offsetof(QCMetrics, number_of_reads), 
     READONLY, "The total amount of reads counted"},
    {NULL},
};

static PyTypeObject QCMetrics_Type = {
    .tp_name = "_qc.QCMetrics",
    .tp_basicsize = sizeof(QCMetrics),
    .tp_dealloc = (destructor)QCMetrics_dealloc,
    .tp_new = (newfunc)QCMetrics__new__,
    .tp_members = QCMetrics_members, 
    .tp_methods = QCMetrics_methods,
};

/* Limit the number of sequences per machine word. So we can statically 
   allocate enough memory per MachineWordPatternMatcher rather than resizing 
   that dynamically as well */
#define MAX_SEQUENCES_PER_WORD 8
#define MAX_SEQUENCE_SIZE 63
/* ASCII only so max index is 127 */
#define BITMASK_INDEX_SIZE 128
typedef uint64_t bitmask_t;

typedef struct AdapterSequenceStruct {
    size_t adapter_index;
    size_t adapter_length;
    bitmask_t found_mask;    
} AdapterSequence; 

typedef struct MachineWordPatternMatcherStruct {
    bitmask_t init_mask;
    bitmask_t found_mask;
    bitmask_t *bitmasks;
    AdapterSequence sequences[MAX_SEQUENCES_PER_WORD];
} MachineWordPatternMatcher;

typedef struct AdapterCounterStruct {
    PyObject_HEAD
    size_t number_of_adapters;
    size_t max_length;
    size_t number_of_sequences;
    counter_t **adapter_counter;
    PyObject *adapters;
    size_t number_of_matchers;
    MachineWordPatternMatcher *matchers;
} AdapterCounter;

static void AdapterCounter_dealloc(AdapterCounter *self) {
    Py_XDECREF(self->adapters);
    for (size_t i=0; i < self->number_of_adapters; i++) {
        PyMem_Free(self->adapter_counter[i]);
    }
    PyMem_Free(self->adapter_counter);
    PyMem_Free(self->matchers);
}

static PyObject *
AdapterCounter__new__(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    static char *kwargnames[] = {"", NULL};
    static char *format = "O:AdapterCounter";
    PyObject *adapter_iterable = NULL;
    PyObject *adapters = NULL; 
    AdapterCounter *self = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, kwargnames, 
        &adapter_iterable)) {
        return NULL;
    } 
    adapters = PySequence_Fast(
        adapter_iterable, 
        "Expected an iterable of adapter sequences");
    if (adapters == NULL) {
        return NULL;
    }
    size_t number_of_adapters = PySequence_Fast_GET_SIZE(adapters);
    for (size_t i=0; i < number_of_adapters; i++) {
        PyObject *adapter = PySequence_Fast_GET_ITEM(adapters, i);
        if (!PyUnicode_CheckExact(adapter)) {
            PyErr_Format(PyExc_TypeError, 
                         "All adapter sequences must be of type str, "
                         "got %s, for %R", Py_TYPE(adapter)->tp_name, adapter);
            goto error;
        }
        if (!PyUnicode_IS_COMPACT_ASCII(adapter)) {
            PyErr_Format(PyExc_ValueError,
                         "Adapter must contain only ASCII characters: %R", 
                         adapter);
            goto error;
        }
        if (PyUnicode_GET_SIZE(adapter) > MAX_SEQUENCE_SIZE) {
            PyErr_Format(PyExc_ValueError, 
                         "Maximum adapter size is %d, got %zd for %R", 
                         MAX_SEQUENCE_SIZE, PyUnicode_GET_SIZE(adapter), adapter);
            goto error;
        }
    }
    AdapterCounter *self = PyObject_New(AdapterCounter, type);

error:
    Py_XDECREF(adapters);
    Py_XDECREF(self);
    return NULL;
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
    SequenceRecord = 
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

    if (PyType_Ready(&QCMetrics_Type) != 0) {
        return NULL;
    }
    Py_INCREF(&QCMetrics_Type);
    if (PyModule_AddObject(m, "QCMetrics", (PyObject *)&QCMetrics_Type) != 0) {
        return NULL;
    }
    PyModule_AddIntConstant(m, "NUMBER_OF_NUCS", NUC_TABLE_SIZE);
    PyModule_AddIntConstant(m, "NUMBER_OF_PHREDS", PHRED_TABLE_SIZE);
    PyModule_AddIntConstant(m, "TABLE_SIZE", PHRED_TABLE_SIZE * NUC_TABLE_SIZE);
    PyModule_AddIntMacro(m, A);
    PyModule_AddIntMacro(m, C);
    PyModule_AddIntMacro(m, G);
    PyModule_AddIntMacro(m, T);
    PyModule_AddIntMacro(m, N);
    PyModule_AddIntMacro(m, PHRED_MAX);
    return m;
}
