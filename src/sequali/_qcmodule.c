/*
Copyright (C) 2023 Leiden University Medical Center
This file is part of sequali

sequali is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

sequali is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with sequali.  If not, see <https://www.gnu.org/licenses/
*/

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "structmember.h"

#include "math.h"
#include "score_to_error_rate.h"

#ifdef __SSE2__
#include "emmintrin.h"
#endif

/* Pointers to types that will be imported in the module initialization section */

static PyTypeObject *PythonArray;  // array.array
static PyTypeObject *SequenceRecord;  // dnaio.SequenceRecord


/*********
 * Utils *
 *********/

static PyObject *
PythonArray_FromBuffer(char typecode, void *buffer, size_t buffersize) 
{
    PyObject *array = PyObject_CallFunction((PyObject *)PythonArray, "C", typecode);
    if (array == NULL) {
        return NULL;
    } 
    /* We cannot paste into the array directly, so use a temporary memoryview */
    PyObject *tmp = PyMemoryView_FromMemory(buffer, buffersize, READONLY);
    if (tmp == NULL) {
        Py_DECREF(array);
        return NULL;
    }
    /* frombytes works in-place, but may return an error. So catch the result. */
    PyObject *result = PyObject_CallMethod(array, "frombytes", "O", tmp);
    Py_DECREF(tmp);
    if (result == NULL) {
        Py_DECREF(array);
        return NULL;
    }
    return array;
}

/*********************
 * FASTQ RECORD VIEW *
 *********************/

/* A structure that holds a pointer to a sequence in an immutable bytes-like
   object. By using only one pointer (8 bytes) and using uint32_t offsets and 
   lengths (4 bytes each) we can make the struct fit on a 64-byte cache line. 

   The idea of FastqRecordView is that it can point to an input buffer instead 
   of copying the information in the buffer. Thus improving memory use, 
   cache locality etc.
*/

typedef struct _FastqRecordViewStruct {
    PyObject_HEAD
    uint8_t *record_start;
    // name_offset is always 1, so no variable needed
    uint32_t name_length;
    uint32_t sequence_offset;
    // Sequence length and qualities length should be the same
    union {
        uint32_t sequence_length;
        uint32_t qualities_length;
    };
    uint32_t qualities_offset;
    PyObject *obj;
} FastqRecordView;

static void 
FastqRecordView_dealloc(FastqRecordView *self)
{
    Py_XDECREF(self->obj);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
FastqRecordView__new__(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyObject *name_obj = NULL; 
    PyObject *sequence_obj = NULL; 
    PyObject *qualities_obj = NULL; 
    static char *kwargnames[] = {"name", "sequence", "qualities", NULL};
    static char *format = "OOO|:FastqRecordView";
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, kwargnames,
        &name_obj, &sequence_obj, &qualities_obj)) {
        return NULL;
    }
    if (!PyUnicode_CheckExact(name_obj)) {
        PyErr_Format(PyExc_TypeError, 
                    "name should be of type str, got %s.", 
                    Py_TYPE(name_obj)->tp_name);
        return NULL;
    }
    if (!PyUnicode_IS_COMPACT_ASCII(name_obj)) {
        PyErr_Format(PyExc_ValueError, 
                     "name should contain only ASCII characters: %R",
                     name_obj);
        return NULL;
    }
    if (!PyUnicode_CheckExact(sequence_obj)) {
        PyErr_Format(PyExc_TypeError, 
                     "sequence should be of type str, got %s.", 
                     Py_TYPE(sequence_obj)->tp_name);
        return NULL;
    }
    if (!PyUnicode_IS_COMPACT_ASCII(sequence_obj)) {
        PyErr_Format(PyExc_ValueError, 
                     "sequence should contain only ASCII characters: %R",
                     sequence_obj);
        return NULL;
    }
    if (!PyUnicode_CheckExact(qualities_obj)) {
        PyErr_Format(PyExc_TypeError, 
                    "qualities should be of type str, got %s.", 
                    Py_TYPE(qualities_obj)->tp_name);
        return NULL;
    }
    if (!PyUnicode_IS_COMPACT_ASCII(qualities_obj)) {
        PyErr_Format(PyExc_ValueError, 
                     "qualities should contain only ASCII characters: %R",
                     qualities_obj); 
        return NULL;
    }
    

    uint8_t *name = PyUnicode_DATA(name_obj);
    size_t name_length = PyUnicode_GET_LENGTH(name_obj);
    uint8_t *sequence = PyUnicode_DATA(sequence_obj);
    size_t sequence_length = PyUnicode_GET_LENGTH(sequence_obj);
    uint8_t *qualities = PyUnicode_DATA(qualities_obj);
    size_t qualities_length = PyUnicode_GET_LENGTH(qualities_obj);

    if (sequence_length != qualities_length) {
        PyErr_Format(
            PyExc_ValueError,
            "sequence and qualities have different lengths: %zd and %zd",
            sequence_length, qualities_length);
        return NULL;
    }

    size_t total_length = name_length + sequence_length + qualities_length + 6;
    if (total_length > UINT32_MAX) {
        // lengths are saved as uint32_t types so throw an error;
        PyErr_Format(
            PyExc_OverflowError, 
            "Total length of FASTQ record exceeds 4 GiB. Record name: %R",
            name_obj);
    }
    PyObject *bytes_obj = PyBytes_FromStringAndSize(NULL, total_length);
    if (bytes_obj == NULL) {
        return PyErr_NoMemory();
    }
    FastqRecordView *self = PyObject_New(FastqRecordView, type);
    if (self == NULL) {
        Py_DECREF(bytes_obj);
        return PyErr_NoMemory();
    }
    uint8_t *buffer = (uint8_t *)PyBytes_AS_STRING(bytes_obj);
    self->record_start = buffer;
    self->name_length = name_length;
    self->sequence_offset = 2 + name_length;
    self->sequence_length = sequence_length;
    self->qualities_offset = 5 + name_length + sequence_length;
    self->obj = bytes_obj;

    buffer[0] = '@';
    memcpy(buffer + 1, name, name_length);
    size_t cursor = 1 + name_length;
    buffer[cursor] = '\n'; cursor +=1;
    memcpy(buffer + cursor, sequence, sequence_length); 
    cursor += sequence_length;
    buffer[cursor] = '\n'; cursor +=1; 
    buffer[cursor] = '+'; cursor += 1;
    buffer[cursor] = '\n'; cursor += 1;
    memcpy(buffer + cursor, qualities, sequence_length);
    cursor += sequence_length; 
    buffer[cursor] = '\n';
    return (PyObject *)self;
}

static PyTypeObject FastqRecordView_Type = {
    .tp_name = "_qc.FastqRecordView",
    .tp_basicsize = sizeof(FastqRecordView),
    .tp_dealloc = (destructor)FastqRecordView_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = (newfunc)FastqRecordView__new__,
};

static inline int 
FastqRecordView_CheckExact(void *obj) 
{
    return Py_TYPE(obj) == &FastqRecordView_Type;
} 


/****************
 * FASTQ PARSER *
 ****************/

typedef struct _FastqParserStruct {
    PyObject_HEAD 
    uint8_t *buffer;
    size_t buffer_size; 
    uint8_t *record_start;
    PyObject *buffer_obj;
    size_t read_in_size;
    PyObject *file_obj;
} FastqParser;

static void
FastqParser_dealloc(FastqParser *self) 
{
    Py_XDECREF(self->buffer_obj);
    Py_XDECREF(self->file_obj);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
FastqParser__new__(PyTypeObject *type, PyObject *args, PyObject *kwargs) 
{
    PyObject *file_obj = NULL;
    size_t read_in_size = 8 * 1024;
    static char *kwargnames[] = {"fileobj", "initial_buffersize", NULL};
    static char *format = "O|n:FastqParser";
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, kwargnames,
        &file_obj, &read_in_size)) {
        return NULL;
    }
    if (read_in_size < 1) {
        PyErr_Format(PyExc_ValueError, 
                     "initial_buffersize must be at least 1, got %zd",
                     read_in_size);
        return NULL;
    }
    PyObject *buffer_obj = PyBytes_FromStringAndSize(NULL, 0);
    if (buffer_obj == NULL) {
        return NULL;
    }
    FastqParser *self = PyObject_New(FastqParser, type);
    if (self == NULL) {
        Py_DECREF(buffer_obj);
        return NULL;
    }
    self->buffer = (uint8_t *)PyBytes_AS_STRING(buffer_obj);
    self->buffer_size = 0;
    self->record_start = self->buffer; 
    self->buffer_obj = buffer_obj;
    self->read_in_size = read_in_size;
    Py_INCREF(file_obj);
    self->file_obj = file_obj;
    return (PyObject *)self;
}

static PyObject *
FastqParser__iter__(PyObject *self)
{
    Py_INCREF(self);
    return self;
}

static PyObject *
FastqParser__next__(FastqParser *self) 
{
    while (1) {
        uint8_t *record_start = self->record_start;
        uint8_t *buffer_end = self->buffer + self->buffer_size;
        while (1) {
            if (record_start + 2 >= buffer_end) {
                break;
            }
            if (record_start[0] != '@') {
                PyErr_Format(
                    PyExc_ValueError,
                    "Record does not start with @ but with %c", 
                    record_start[0]
                );
                return NULL;
            }
            uint8_t *name_end = memchr(record_start, '\n', 
                                       buffer_end - record_start);
            if (name_end == NULL) {
                break;
            }
            size_t name_length = name_end - (record_start + 1);
            uint8_t *sequence_start = name_end + 1;
            uint8_t *sequence_end = memchr(sequence_start, '\n', 
                                           buffer_end - sequence_start);
            if (sequence_end == NULL) {
                break;
            }
            size_t sequence_length = sequence_end - sequence_start; 
            uint8_t *second_header_start = sequence_end + 1; 
            if ((second_header_start < buffer_end) && second_header_start[0] != '+') {
                PyErr_Format(
                    PyExc_ValueError,
                    "Record second header does not start with + but with %c",
                    second_header_start[0]
                );
                return NULL;
            }
            uint8_t *second_header_end = memchr(second_header_start, '\n', 
                                               buffer_end - second_header_start);
            if (second_header_end == NULL) {
                break;
            }
            uint8_t *qualities_start = second_header_end + 1;
            uint8_t *qualities_end = memchr(qualities_start, '\n',
                                            buffer_end - qualities_start);
            if (qualities_end == NULL) {
                break;
            }
            size_t qualities_length = qualities_end - qualities_start;
            if (sequence_length != qualities_length) {
                PyErr_Format(
                    PyExc_ValueError,
                    "Record sequence and qualities do not have equal length, %R",
                    PyUnicode_DecodeASCII((char *)record_start + 1, name_length, NULL)
                );
            }
            FastqRecordView *record = PyObject_New(FastqRecordView, &FastqRecordView_Type);
            if (record == NULL) {
                return NULL;
            }
            record->record_start = record_start;
            record->name_length = name_length;
            record->sequence_offset = sequence_start - record_start;
            record->sequence_length = sequence_length;
            record->qualities_offset = qualities_start - record_start;
            Py_INCREF(self->buffer_obj);
            record->obj = self->buffer_obj;
            self->record_start = qualities_end + 1;
            return (PyObject *)record;
        }
        if (record_start == self->buffer && self->buffer_size != 0) {
            // The FASTQ record is bigger than the current buffer_size
            self->read_in_size *= 2;
        }
        size_t bytes_left = buffer_end - record_start;
        size_t read_in_size = self->read_in_size - bytes_left;
        PyObject *new_bytes = PyObject_CallMethod(self->file_obj, "read", "n", read_in_size);
        if (new_bytes == NULL) {
            return NULL;
        }
        if (!PyBytes_CheckExact(new_bytes)) {
            PyErr_Format(
                PyExc_TypeError,
                "file_obj %R is not a binary IO type, got %s", 
                self->file_obj, Py_TYPE(self->file_obj)->tp_name
            );
            return NULL;
        }
        size_t new_bytes_size = PyBytes_GET_SIZE(new_bytes);
        size_t new_buffer_size = bytes_left + new_bytes_size;
        if (new_buffer_size == 0) {
            // Entire file is read
            PyErr_SetNone(PyExc_StopIteration);
            return NULL;
        } else if (new_bytes_size == 0) {
            // Incomplete record at the end of file;
            PyErr_Format(
                PyExc_ValueError, 
                "Incomplete record at the end of file %s", 
                self->record_start);
            return NULL;
        }
        PyObject *new_buffer_obj = PyBytes_FromStringAndSize(NULL, new_buffer_size);
        if (new_buffer_obj == NULL) {
            Py_DECREF(new_bytes);
            return NULL;
        }
        uint8_t *new_buffer = (uint8_t *)PyBytes_AS_STRING(new_buffer_obj);
        memcpy(new_buffer, record_start, bytes_left);
        memcpy(new_buffer + bytes_left, PyBytes_AS_STRING(new_bytes), new_bytes_size);
        Py_DECREF(new_bytes);
        PyObject *tmp = self->buffer_obj;
        self->buffer_obj = new_buffer_obj;
        self->buffer = new_buffer;
        self->buffer_size = new_buffer_size;
        self->record_start = new_buffer;
        Py_DECREF(tmp);
    }
}

PyTypeObject FastqParser_Type = {
    .tp_name = "_qc.FastqParser",
    .tp_basicsize = sizeof(FastqParser),
    .tp_dealloc = (destructor)FastqParser_dealloc,
    .tp_new = FastqParser__new__,
    .tp_iter = (iternextfunc)FastqParser__iter__,
    .tp_iternext = (iternextfunc)FastqParser__next__,
};

/**************
 * QC METRICS *
 **************/

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
    uint8_t phred_offset;
    counttable_t *count_tables;
    Py_ssize_t max_length;
    size_t number_of_reads;
    counter_t gc_content[101];
    counter_t phred_scores[PHRED_MAX + 1];
} QCMetrics;

static void
QCMetrics_dealloc(QCMetrics *self) {
    PyMem_Free(self->count_tables);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
QCMetrics__new__(PyTypeObject *type, PyObject *args, PyObject *kwargs){
    static char *kwargnames[] = {NULL};
    static char *format = ":QCMetrics";
    uint8_t phred_offset = 33;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, kwargnames)) {
        return NULL;
    }
    QCMetrics *self = PyObject_New(QCMetrics, type);
    self->max_length = 0;
    self->phred_offset = phred_offset;
    self->count_tables = NULL;
    self->number_of_reads = 0;
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
"    A FastqRecordView object.\n"
);

#define QCMetrics_add_read_method METH_O

static PyObject * 
QCMetrics_add_read(QCMetrics *self, FastqRecordView *read) 
{
    if (!FastqRecordView_CheckExact(read)) {
        PyErr_Format(PyExc_TypeError, 
                     "read should be a FastqRecordView object, got %s", 
                     Py_TYPE(read)->tp_name);
        return NULL;
    }
    const uint8_t *record_start = read->record_start;
    size_t sequence_length = read->sequence_length;
    const uint8_t *sequence = record_start + read->sequence_offset;
    const uint8_t *qualities = record_start + read->qualities_offset;
    uint8_t phred_offset = self->phred_offset;
    counter_t base_counts[NUC_TABLE_SIZE] = {0, 0, 0, 0, 0};
    double accumulated_error_rate = 0.0;

    if (sequence_length > self->max_length) {
        if (QCMetrics_resize(self, sequence_length) != 0) {
            return NULL;
        }
    }

    self->number_of_reads += 1; 
    for (size_t i=0; i < (size_t)sequence_length; i+=1) {
        uint8_t c = sequence[i];
        uint8_t q = qualities[i] - phred_offset;
        if (q > PHRED_MAX) {
            PyErr_Format(
                PyExc_ValueError, 
                "Not a valid phred character: %c", qualities[i]
            );
            return NULL;
        }
        uint8_t q_index = phred_to_index(q);
        uint8_t c_index = NUCLEOTIDE_TO_INDEX[c];
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
    Py_RETURN_NONE;
}

PyDoc_STRVAR(QCMetrics_count_table__doc__,
"count_table($self, /)\n"
"--\n"
"\n"
"Return a array.array on the produced count table. \n"
);

#define QCMetrics_count_table_method METH_NOARGS

static PyObject *
QCMetrics_count_table(QCMetrics *self, PyObject *Py_UNUSED(ignore))
{
    return PythonArray_FromBuffer(
        'Q', 
        self->count_tables, 
        self->max_length *sizeof(counttable_t));
}

PyDoc_STRVAR(QCMetrics_gc_content__doc__,
"gc_content($self, /)\n"
"--\n"
"\n"
"Return a array.array on the produced gc content counts. \n"
);

#define QCMetrics_gc_content_method METH_NOARGS

static PyObject *
QCMetrics_gc_content(QCMetrics *self, PyObject *Py_UNUSED(ignore))
{
    return PythonArray_FromBuffer(
        'Q',
        self->gc_content,
        sizeof(self->gc_content)
    );
}

PyDoc_STRVAR(QCMetrics_phred_scores__doc__,
"phred_scores($self, /)\n"
"--\n"
"\n"
"Return a array.array on the produced average phred score counts. \n"
);

#define QCMetrics_phred_scores_method METH_NOARGS

static PyObject *
QCMetrics_phred_scores(QCMetrics *self, PyObject *Py_UNUSED(ignore))
{
    return PythonArray_FromBuffer(
        'Q',
        self->phred_scores,
        sizeof(self->phred_scores)
    );
}

static PyMethodDef QCMetrics_methods[] = {
    {"add_read",  (PyCFunction)QCMetrics_add_read, 
     QCMetrics_add_read_method,  QCMetrics_add_read__doc__},
    {"count_table", (PyCFunction)QCMetrics_count_table, 
     QCMetrics_count_table_method, QCMetrics_count_table__doc__},
    {"gc_content", (PyCFunction)QCMetrics_gc_content, 
     QCMetrics_gc_content_method, QCMetrics_gc_content__doc__},
    {"phred_scores", (PyCFunction)QCMetrics_phred_scores, 
     QCMetrics_phred_scores_method, QCMetrics_phred_scores__doc__},
    {NULL},
};

static PyMemberDef QCMetrics_members[] = {
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


/*******************
 * ADAPTER COUNTER *
 *******************/

typedef uint64_t bitmask_t;
#define MACHINE_WORD_BITS (sizeof(bitmask_t) * 8)
#define MAX_SEQUENCE_SIZE MACHINE_WORD_BITS

typedef struct AdapterSequenceStruct {
    size_t adapter_index;
    size_t adapter_length;
    bitmask_t found_mask;    
} AdapterSequence; 

/* Because we use NUCLEOTIDE_TO_INDEX we can save the bitmasks in the struct
   itself. There are only 5 nucleotides (ACGTN) so this uses 40 bytes. With
   init_mask and found_mask costing 8 bytes each the entire struct up to
   number of sequences fits on one cache line of 64 bytes. Except for the 
   sequences pointer, but that is only used in case of a match. That makes 
   accessing the bitmasks very quick memorywise. */
typedef struct MachineWordPatternMatcherStruct {
    bitmask_t init_mask;
    bitmask_t found_mask;
    bitmask_t bitmasks[NUC_TABLE_SIZE];
    size_t number_of_sequences;
    AdapterSequence *sequences;
} MachineWordPatternMatcher;

static void 
MachineWordPatternMatcher_destroy(MachineWordPatternMatcher *matcher) {
    PyMem_Free(matcher->sequences);
    matcher->sequences = NULL;
}

#ifdef __SSE2__
typedef struct AdapterSequenceSSE2Struct {
    size_t adapter_index;
    size_t adapter_length;
    __m128i found_mask;
} AdapterSequenceSSE2; 

typedef struct MachineWordPatternMatcherSSE2Struct {
    __m128i init_mask;
    __m128i found_mask;
    __m128i bitmasks[NUC_TABLE_SIZE];
    size_t number_of_sequences;
    AdapterSequenceSSE2 *sequences;
} MachineWordPatternMatcherSSE2;

static void 
MachineWordPatternMatcherSSE2_destroy(MachineWordPatternMatcherSSE2 *matcher) {
    PyMem_Free(matcher->sequences);
}
#endif


typedef struct AdapterCounterStruct {
    PyObject_HEAD
    size_t number_of_adapters;
    size_t max_length;
    size_t number_of_sequences;
    counter_t **adapter_counter;
    PyObject *adapters;
    size_t number_of_matchers;
    MachineWordPatternMatcher *matchers;
#ifdef __SSE2__
    size_t number_of_sse2_matchers;
    MachineWordPatternMatcherSSE2 *sse2_matchers;
#endif
} AdapterCounter;

static void AdapterCounter_dealloc(AdapterCounter *self) {
    Py_XDECREF(self->adapters);
    if (self->adapter_counter != NULL) {
        for (size_t i=0; i < self->number_of_adapters; i++) {
            PyMem_Free(self->adapter_counter[i]);
        }
    }
    PyMem_Free(self->adapter_counter);
    for (size_t i=0; i < self->number_of_matchers; i++) {
        MachineWordPatternMatcher_destroy(self->matchers + i);
    }
    PyMem_Free(self->matchers);
    
    #ifdef __SSE2__
    for (size_t i=0; i < self->number_of_sse2_matchers; i++) {
        MachineWordPatternMatcherSSE2_destroy(self->sse2_matchers + i);
    }
    PyMem_Free(self->sse2_matchers);
    #endif
    Py_TYPE(self)->tp_free((PyObject *)self);
}

#ifdef __SSE2__
int AdapterCounter_SSE2_convert(AdapterCounter *self) {
    self->number_of_sse2_matchers = self->number_of_matchers / 2;
    if (self->number_of_sse2_matchers == 0) {
        return 0;
    } 
    MachineWordPatternMatcherSSE2 *tmp = PyMem_Malloc(
        self->number_of_sse2_matchers * sizeof(MachineWordPatternMatcherSSE2));
    if (tmp == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    self->sse2_matchers = tmp;
    memset(self->sse2_matchers, 0, self->number_of_sse2_matchers * sizeof(MachineWordPatternMatcherSSE2));
    for (size_t i=0; i < self->number_of_sse2_matchers; i++) {
        MachineWordPatternMatcherSSE2 *sse2_matcher = self->sse2_matchers + i; 
        MachineWordPatternMatcher *normal_matcher1 = self->matchers + (i * 2);
        MachineWordPatternMatcher *normal_matcher2 = self->matchers + (i * 2 + 1);
        sse2_matcher->init_mask = _mm_set_epi64x(normal_matcher1->init_mask, 
                                                 normal_matcher2->init_mask);
        sse2_matcher->found_mask = _mm_set_epi64x(normal_matcher1->found_mask, 
                                                  normal_matcher2->found_mask);
        for (size_t j=0; j < NUC_TABLE_SIZE; j++) {
            sse2_matcher->bitmasks[j] = _mm_set_epi64x(
                normal_matcher1->bitmasks[j], normal_matcher2->bitmasks[j]);
        }
        sse2_matcher->number_of_sequences = normal_matcher1->number_of_sequences + normal_matcher2->number_of_sequences;
        AdapterSequenceSSE2 *seq_tmp = PyMem_Malloc(sse2_matcher->number_of_sequences * sizeof(AdapterSequenceSSE2));
        if (seq_tmp == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        sse2_matcher->sequences = seq_tmp;
        for (size_t j = 0; j < normal_matcher1->number_of_sequences; j++) {
            AdapterSequenceSSE2 *sse2_adapter = sse2_matcher->sequences + j;
            AdapterSequence *normal_adapter = normal_matcher1->sequences + j; 
            sse2_adapter->adapter_index = normal_adapter->adapter_index;
            sse2_adapter->adapter_length = normal_adapter->adapter_length;
            sse2_adapter->found_mask = _mm_set_epi64x(normal_adapter->found_mask, 0);
        };
        for (size_t j = 0; j < normal_matcher2->number_of_sequences; j++) {
            size_t offset = normal_matcher1->number_of_sequences;
            AdapterSequenceSSE2 *sse2_adapter = sse2_matcher->sequences + j + offset;
            AdapterSequence *normal_adapter = normal_matcher2->sequences + j; 
            sse2_adapter->adapter_index = normal_adapter->adapter_index;
            sse2_adapter->adapter_length = normal_adapter->adapter_length;
            sse2_adapter->found_mask = _mm_set_epi64x(0, normal_adapter->found_mask);
        };
    }

    for (size_t i=0; i<(self->number_of_sse2_matchers * 2); i++) {
        MachineWordPatternMatcher_destroy(self->matchers + i);
    }
    size_t number_of_remaining_matchers = self->number_of_matchers % 2;
    if (number_of_remaining_matchers == 0) {
        self->number_of_matchers = 0;
        PyMem_FREE(self->matchers);
        self->matchers = NULL;
        return 0;
    }
    MachineWordPatternMatcher *matcher_tmp = PyMem_Malloc(sizeof(MachineWordPatternMatcher));
    if (matcher_tmp == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    memcpy(matcher_tmp, self->matchers + (self->number_of_sse2_matchers * 2), sizeof(MachineWordPatternMatcher));
    PyMem_FREE(self->matchers);
    self->matchers = matcher_tmp;
    self->number_of_matchers = 1;
    return 0;
}
#endif

static void
populate_bitmask(bitmask_t *bitmask, char *word, size_t word_length) 
{
    for (size_t i=0; i < word_length; i++) {
        char c = word[i];
        if (c == 0) {
            continue;
        }
        uint8_t index = NUCLEOTIDE_TO_INDEX[(uint8_t)c];
        /* Match both upper and lowercase */
        bitmask[index] |= (bitmask_t)1ULL << i;
    }
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
    adapters = PySequence_Tuple(adapter_iterable);
    if (adapters == NULL) {
        return NULL;
    }
    size_t number_of_adapters = PyTuple_GET_SIZE(adapters);
    if (number_of_adapters < 1) {
        PyErr_SetString(PyExc_ValueError, "At least one adapter is expected");
        goto error;
    } 
    for (size_t i=0; i < number_of_adapters; i++) {
        PyObject *adapter = PyTuple_GET_ITEM(adapters, i);
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
        if ((size_t)PyUnicode_GET_LENGTH(adapter) > MAX_SEQUENCE_SIZE) {
            PyErr_Format(PyExc_ValueError, 
                         "Maximum adapter size is %d, got %zd for %R", 
                         MAX_SEQUENCE_SIZE, PyUnicode_GET_LENGTH(adapter), adapter);
            goto error;
        }
    }
    self = PyObject_New(AdapterCounter, type);
    counter_t **counter_tmp = PyMem_Malloc(sizeof(counter_t *) * number_of_adapters);
    if (counter_tmp == NULL) {
        PyErr_NoMemory();
        goto error;
    }
    memset(counter_tmp, 0, sizeof(counter_t *) * number_of_adapters);
    self->adapter_counter = counter_tmp;
    self->adapters = NULL;
    self->matchers = NULL;
    self->max_length = 0;
    self->number_of_adapters = number_of_adapters;
    self->number_of_matchers = 0;
    self->number_of_sequences = 0;
    #ifdef __SSE2__ 
    self->number_of_sse2_matchers = 0;
    self->sse2_matchers = NULL;
    #endif
    size_t adapter_index = 0;
    size_t matcher_index = 0;
    PyObject *adapter;
    Py_ssize_t adapter_length;
    char machine_word[MACHINE_WORD_BITS];
    matcher_index = 0;
    while(adapter_index < number_of_adapters) {
        self->number_of_matchers += 1; 
        MachineWordPatternMatcher *tmp = PyMem_Realloc(
            self->matchers, sizeof(MachineWordPatternMatcher) * self->number_of_matchers);
        if (tmp == NULL) {
            PyErr_NoMemory();
            goto error;
        }
        self->matchers = tmp;
        memset(self->matchers + matcher_index, 0, sizeof(MachineWordPatternMatcher));
        bitmask_t found_mask = 0;
        bitmask_t init_mask = 0;
        size_t adapter_in_word_index = 0; 
        size_t word_index = 0;
        MachineWordPatternMatcher *matcher = self->matchers + matcher_index;
        memset(machine_word, 0, MACHINE_WORD_BITS);
        while (adapter_index < number_of_adapters) {
            adapter = PyTuple_GET_ITEM(adapters, adapter_index); 
            adapter_length = PyUnicode_GET_LENGTH(adapter);
            if ((word_index + adapter_length) > MACHINE_WORD_BITS) {
                break;
            }
            memcpy(machine_word + word_index, PyUnicode_DATA(adapter), adapter_length);
            init_mask |= (1ULL << word_index);
            word_index += adapter_length;
            AdapterSequence adapter_sequence = {
                .adapter_index = adapter_index,
                .adapter_length = adapter_length,
                .found_mask = 1ULL << (word_index - 1),  /* Last character */
            };
            AdapterSequence *adapt_tmp = PyMem_Realloc(matcher->sequences, (adapter_in_word_index + 1) * sizeof(AdapterSequence)); 
            if (adapt_tmp == NULL) {
                PyErr_NoMemory();
                goto error;
            }
            matcher->sequences = adapt_tmp;
            matcher->sequences[adapter_in_word_index] = adapter_sequence;
            found_mask |= adapter_sequence.found_mask;
            adapter_in_word_index += 1;
            adapter_index += 1;
        }
        populate_bitmask(matcher->bitmasks, machine_word, word_index);
        matcher->found_mask = found_mask;
        matcher->init_mask = init_mask;
        matcher->number_of_sequences = adapter_in_word_index;
        matcher_index += 1;
    }
    self->adapters = adapters;
    #ifdef __SSE2__
    if (AdapterCounter_SSE2_convert(self) != 0) {
        return NULL;
    }
    #endif
    return (PyObject *)self;

error:
    Py_XDECREF(adapters);
    Py_XDECREF(self);
    return NULL;
}

static int 
AdapterCounter_resize(AdapterCounter *self, size_t new_size)
{
    if (self->max_length >= new_size) {
        return 0;
    }
    size_t old_size = self->max_length;
    for (size_t i=0; i < self->number_of_adapters; i++) {
        counter_t *tmp = PyMem_Realloc(self->adapter_counter[i],
                                       new_size * sizeof(counter_t));
        if (tmp == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        self->adapter_counter[i] = tmp;
        memset(self->adapter_counter[i] + old_size, 0,
               (new_size - old_size) * sizeof(counter_t));
    }
    self->max_length = new_size;
    return 0;
}

#ifdef __SSE2__
static inline int bitwise_and_nonzero_si128(__m128i vector1, __m128i vector2) {
    /* There is no way to directly check if an entire vector is set to 0
       so some trickery needs to be done to ascertain if one of the bits is
       set.
       _mm_movemask_epi8 only catches the most significant bit. So we need to
       set that bit. Comparison for larger than 0 does not work since only
       signed comparisons are available. So the most significant bit makes
       integers smaller than 0. Instead we do a saturated add of 127.
       _mm_adds_epu8 works on unsigned integers. So 0b10000000 (128) will become
       255. Also everything above 0 will trigger the last bit to be set. 0
       itself results in 0b01111111 so the most significant bit will not be
       set.
       The sequence of instructions below is faster than 
       return (!_mm_test_all_zeros(vector1, vector2)); 
       which is available in SSE4.1. So there is no value in moving up one
       instruction set. */
    __m128i and = _mm_and_si128(vector1, vector2);
    __m128i res = _mm_adds_epu8(and, _mm_set1_epi8(127));
    return _mm_movemask_epi8(res);
}
#endif

PyDoc_STRVAR(AdapterCounter_add_read__doc__,
"add_read($self, read, /)\n"
"--\n"
"\n"
"Add a read to the adapter counter. \n"
"\n"
"  read\n"
"    A FastqRecordView object.\n"
);

#define AdapterCounter_add_read_method METH_O

static PyObject *
AdapterCounter_add_read(AdapterCounter *self, FastqRecordView *read) 
{
    if (!FastqRecordView_CheckExact(read)) {
        PyErr_Format(PyExc_TypeError, 
                     "read should be a FastqRecordView object, got %s", 
                     Py_TYPE(read)->tp_name);
        return NULL;
    }
    self->number_of_sequences += 1;
    uint8_t *sequence = read->record_start + read->sequence_offset;
    size_t sequence_length = read->sequence_length;

    if (sequence_length > self->max_length) {
        int ret = AdapterCounter_resize(self, sequence_length);
        if (ret != 0) {
            return NULL;
        }
    }

    for (size_t i=0; i<self->number_of_matchers; i++){
        MachineWordPatternMatcher *matcher = self->matchers + i;
        bitmask_t found_mask = matcher->found_mask;
        bitmask_t init_mask = matcher->init_mask;
        bitmask_t R = 0;
        bitmask_t *bitmask = matcher->bitmasks;
        bitmask_t already_found = 0;
        for (size_t j=0; j<sequence_length; j++) {
            R <<= 1;
            R |= init_mask;
            uint8_t index = NUCLEOTIDE_TO_INDEX[sequence[j]];
            R &= bitmask[index];
            if (R & found_mask) {
                /* Check which adapter was found */
                size_t number_of_adapters = matcher->number_of_sequences;
                for (size_t k=0; k < number_of_adapters; k++) {
                    AdapterSequence *adapter = matcher->sequences + k;
                    bitmask_t adapter_found_mask = adapter->found_mask;
                    if (adapter_found_mask & already_found) {
                        continue;
                    }
                    if (R & adapter_found_mask) {
                        size_t found_position = j - adapter->adapter_length + 1;
                        self->adapter_counter[adapter->adapter_index][found_position] += 1;
                        // Make sure we only find the adapter once at the earliest position;
                        already_found |= adapter_found_mask;
                    }
                }
            }
        }
    }
    #ifdef __SSE2__
    for (size_t i=0; i<self->number_of_sse2_matchers; i++){
        MachineWordPatternMatcherSSE2 *matcher = self->sse2_matchers + i;
        __m128i found_mask = matcher->found_mask;
        __m128i init_mask = matcher->init_mask;
        __m128i R = _mm_setzero_si128();
        __m128i *bitmask = matcher->bitmasks;
        __m128i already_found = _mm_setzero_si128();

        for (size_t j=0; j<sequence_length; j++) {
            R = _mm_slli_epi64(R, 1);
            R = _mm_or_si128(R, init_mask);
            uint8_t index = NUCLEOTIDE_TO_INDEX[sequence[j]];
            __m128i mask = bitmask[index];
            R = _mm_and_si128(R, mask);
            if (bitwise_and_nonzero_si128(R, found_mask)) {
                /* Check which adapter was found */
                size_t number_of_adapters = matcher->number_of_sequences;
                for (size_t k=0; k < number_of_adapters; k++) {
                    AdapterSequenceSSE2 *adapter = matcher->sequences + k;
                    __m128i adapter_found_mask = adapter->found_mask;
                    if (bitwise_and_nonzero_si128(adapter_found_mask, already_found)) {
                        continue;
                    }
                    if (bitwise_and_nonzero_si128(R, adapter_found_mask)) {
                        size_t found_position = j - adapter->adapter_length + 1;
                        self->adapter_counter[adapter->adapter_index][found_position] += 1;
                        // Make sure we only find the adapter once at the earliest position;
                        already_found = _mm_or_si128(already_found, adapter_found_mask);
                    }
                }
            }
        }
    }
    #endif
    Py_RETURN_NONE;
}

PyDoc_STRVAR(AdapterCounter_get_counts__doc__,
"get_counts($self, /)\n"
"--\n"
"\n"
"Return the counts as a list of tuples. Each tuple contains the adapter, \n"
"and an array.array counts per position. \n"
);

# define AdapterCounter_get_counts_method METH_NOARGS

static PyObject *
AdapterCounter_get_counts(AdapterCounter *self, PyObject *Py_UNUSED(ignore))
{
    if (self->number_of_sequences < 1) {
        PyErr_SetString(PyExc_ValueError, "No sequences were counted yet.");
        return NULL;
    }
    PyObject *counts_list = PyList_New(self->number_of_adapters);
    if (counts_list == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    for (size_t i=0; i < self->number_of_adapters; i++) {
        PyObject *tup = PyTuple_New(2);
        PyObject *counts = PythonArray_FromBuffer(
            'Q',
            self->adapter_counter[i],
            self->max_length * sizeof(counter_t)
        );
        if (counts == NULL) {
            return NULL;
        }
        PyObject *adapter = PyTuple_GET_ITEM(self->adapters, i);
        Py_INCREF(adapter);
        PyTuple_SET_ITEM(tup, 0, adapter);
        PyTuple_SET_ITEM(tup, 1, counts);
        PyList_SET_ITEM(counts_list, i, tup);
    }
    return counts_list;
}


static PyMethodDef AdapterCounter_methods[] = {
    {"add_read", (PyCFunction)AdapterCounter_add_read,
     AdapterCounter_add_read_method, AdapterCounter_add_read__doc__},
    {"get_counts", (PyCFunction)AdapterCounter_get_counts, 
     AdapterCounter_get_counts_method, AdapterCounter_get_counts__doc__},
    {NULL},
};

static PyMemberDef AdapterCounter_members[] = {
    {"max_length", T_ULONGLONG, offsetof(AdapterCounter, max_length), READONLY, 
    "The length of the longest read"},
    {"number_of_sequences", T_ULONGLONG, 
     offsetof(AdapterCounter, number_of_sequences), READONLY, 
     "The total counted number of sequences"},
    {"adapters", T_OBJECT_EX, offsetof(AdapterCounter, adapters), READONLY, 
     "The adapters that are searched for"},
    {NULL},
};

static PyTypeObject AdapterCounter_Type = {
    .tp_name = "_qc.AdapterCounter",
    .tp_basicsize = sizeof(AdapterCounter),
    .tp_dealloc = (destructor)AdapterCounter_dealloc,
    .tp_new = (newfunc)AdapterCounter__new__, 
    .tp_members = AdapterCounter_members,
    .tp_methods = AdapterCounter_methods,
};


/********************
 * Per Tile Quality *
 ********************/

typedef struct _BaseQualityStruct {
    counter_t total_bases;
    double total_error;  /* double for now, fixed point might be better */ 
} BaseQuality;

typedef struct _PerTileQualityStruct {
    PyObject_HEAD
    uint8_t phred_offset;
    char skipped;
    BaseQuality **base_qualities;
    size_t number_of_tiles;
    size_t max_length;
    size_t number_of_reads;
    PyObject *skipped_reason;
} PerTileQuality;

static void
PerTileQuality_dealloc(PerTileQuality *self) {
    Py_XDECREF(self->skipped_reason);
    for (size_t i=0; i < self->number_of_tiles; i++) {
        BaseQuality *tile_quals = self->base_qualities[i];
        PyMem_Free(tile_quals);
    }
    PyMem_Free(self->base_qualities);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
PerTileQuality__new__(PyTypeObject *type, PyObject *args, PyObject *kwargs){
    static char *kwargnames[] = {NULL};
    static char *format = ":PerTileQuality";
    uint8_t phred_offset = 33;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, kwargnames)) {
        return NULL;
    }
    PerTileQuality *self = PyObject_New(PerTileQuality, type);
    self->max_length = 0;
    self->phred_offset = phred_offset;
    self->base_qualities = NULL;
    self->number_of_reads = 0;
    self->number_of_tiles = 0;
    self->skipped = 0;
    self->skipped_reason = NULL;
    return (PyObject *)self;
}

static int 
PerTileQuality_resize_tile_array(PerTileQuality *self, size_t highest_tile) 
{
    if (highest_tile < self->number_of_tiles) {
        return 0;
    }
    BaseQuality **new_qualites = PyMem_Realloc(
        self->base_qualities, highest_tile * sizeof(BaseQuality *));
    if (new_qualites == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    size_t previous_number_of_tiles = self->number_of_tiles;
    memset(new_qualites + previous_number_of_tiles, 0, 
           (highest_tile - previous_number_of_tiles) * sizeof(BaseQuality *));
    self->base_qualities = new_qualites;
    self->number_of_tiles = highest_tile;
    return 0;
};

static int
PerTileQuality_resize_tiles(PerTileQuality *self, size_t new_length) 
{
    if (new_length < self->max_length) {
        return 0;
    }
    BaseQuality **base_qualities = self->base_qualities;
    size_t number_of_tiles = self->number_of_tiles; 
    size_t old_length = self->max_length;
    for (size_t i=0; i<number_of_tiles; i++) {
        BaseQuality *tile_quals = base_qualities[i];
        if (tile_quals == NULL) {
            continue;
        }
        BaseQuality *tmp_quals = PyMem_Realloc(tile_quals, new_length * sizeof(BaseQuality));
        if (tmp_quals == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        memset(tmp_quals + old_length, 0, (new_length - old_length) * sizeof(BaseQuality));
        base_qualities[i] = tmp_quals;
    }
    self->max_length = new_length;
    return 0;
}

/**
 * @brief Parse illumina header and return the tile ID
 * 
 * @param header A string pointing to the header
 * @param header_length length of the header string
 * @return long the tile_id or -1 if there was a parse error.
 */
static
long illumina_header_to_tile_id(const char *header, size_t header_length) {

    /* The following link contains the header format:
       https://support.illumina.com/help/BaseSpace_OLH_009008/Content/Source/Informatics/BS/FileFormat_FASTQ-files_swBS.htm
       It reports the following format:
       @<instrument>:<run number>:<flowcell ID>:<lane>:<tile>:<x-pos>:<y-pos>:<UMI> <read>:<is filtered>:<control number>:<index>
       The tile ID is after the fourth colon.
    */
    size_t colon_count = 0;
    size_t tile_number_offset = -1; 
    for (size_t i=0; i < header_length; i++) {
        if (header[i] == ':') {
            colon_count += 1;
            if (colon_count == 4) {
                tile_number_offset = i + 1;
                break;
            }
        }
    }
    if (colon_count != 4) {
        return -1;
    }
    const char *tile_start = header + tile_number_offset;
    char *tile_end = NULL;
    long tile_id = strtol(tile_start, &tile_end, 10);
    /* tile_end must point to a colon (the one before x-pos) after tile_start */
    if (tile_end == NULL || tile_end == tile_start || *tile_end != ':') {
        errno = 0;  /* Clear errno because there might be a parse error set by strtol*/
        return -1;
    }
    return tile_id;
}

PyDoc_STRVAR(PerTileQuality_add_read__doc__,
"add_read($self, read, /)\n"
"--\n"
"\n"
"Add a read to the PerTileQuality Metrics. \n"
"\n"
"  read\n"
"    A dnaio.SequenceRecord object.\n"
);

#define PerTileQuality_add_read_method METH_O

static PyObject *
PerTileQuality_add_read(PerTileQuality *self, FastqRecordView *read)
{
    if (self->skipped) {
        Py_RETURN_NONE;
    }
    if (!FastqRecordView_CheckExact(read)) {
        PyErr_Format(PyExc_TypeError, 
                     "read should be a FastqRecordView object, got %s", 
                     Py_TYPE(read)->tp_name);
        return NULL;
    }
    uint8_t *record_start = read->record_start;
    const char *header = (char *)(record_start + 1);
    size_t header_length = read->name_length;
    const uint8_t *qualities = record_start + read->qualities_offset;
    size_t sequence_length = read->sequence_length;
    uint8_t phred_offset = self->phred_offset;

    long tile_id = illumina_header_to_tile_id(header, header_length);
    if (tile_id == -1) {
        self->skipped_reason = PyUnicode_FromFormat(
            "Can not parse header: %s", header); 
        self->skipped = 1;
        Py_RETURN_NONE;
    }

    if (sequence_length > self->max_length) {
        if (PerTileQuality_resize_tiles(self, sequence_length) != 0) {
            return NULL;
        }
    }

    /* Tile index must be one less than the highest number of tiles otherwise 
       the index is not in the tile array. */
    if (((size_t)tile_id + 1) > self->number_of_tiles) {
        if (PerTileQuality_resize_tile_array(self, tile_id + 1) != 0) {
            return NULL;
        }
    }
    
    BaseQuality *tile_qualities = self->base_qualities[tile_id];
    if (tile_qualities == NULL) {
        tile_qualities = PyMem_Malloc(self->max_length * sizeof(BaseQuality));
        if (tile_qualities == NULL) {
            PyErr_NoMemory();
            return NULL;
        }
        memset(tile_qualities, 0, self->max_length * sizeof(BaseQuality));
        self->base_qualities[tile_id] = tile_qualities;
    }

    self->number_of_reads += 1;
    for (size_t i=0; i < sequence_length; i+=1) {
        uint8_t q = qualities[i] - phred_offset;
        if (q > PHRED_MAX) {
            PyErr_Format(
                PyExc_ValueError,
                "Not a valid phred character: %c", qualities[i]
            );
            return NULL;
        }
        tile_qualities[i].total_bases += 1;
        tile_qualities[i].total_error += SCORE_TO_ERROR_RATE[q];
    }
    Py_RETURN_NONE;
}


PyDoc_STRVAR(PerTileQuality_get_tile_averages__doc__,
"add_read($self, /)\n"
"--\n"
"\n"
"Get a list of tuples with the tile IDs and a list of their averages. \n"
);

#define PerTileQuality_get_tile_averages_method METH_NOARGS

static PyObject *
PerTileQuality_get_tile_averages(PerTileQuality *self, PyObject *Py_UNUSED(ignore))
{
    BaseQuality **base_qualities = self->base_qualities;
    size_t maximum_tile = self->number_of_tiles;
    size_t tile_length = self->max_length;
    PyObject *result = PyList_New(0);
    if (result == NULL) {
        return PyErr_NoMemory();
    }

    for (size_t i=0; i<maximum_tile; i++) {
        BaseQuality *quals = base_qualities[i];
        if (quals == NULL) {
            continue;
        }
        PyObject *entry = PyTuple_New(2);
        PyObject *tile_id = PyLong_FromSize_t(i);
        PyObject *averages_list = PyList_New(tile_length);
        if (entry == NULL || tile_id == NULL || averages_list == NULL) {
            Py_DECREF(result);
            return PyErr_NoMemory();
        }
        for (size_t j=0; j<tile_length; j++) {
            BaseQuality qual_entry = quals[j];
            double average = qual_entry.total_error / 
                ((double)qual_entry.total_bases);
            PyObject *average_obj = PyFloat_FromDouble(average);
            if (average_obj == NULL) {
                Py_DECREF(result);
                return PyErr_NoMemory();
            }
            PyList_SET_ITEM(averages_list, j, average_obj);
        }
        PyTuple_SET_ITEM(entry, 0, tile_id);
        PyTuple_SET_ITEM(entry, 1, averages_list);
        int ret = PyList_Append(result, entry);
        if (ret != 0) {
            Py_DECREF(result);
            return NULL;
        }
        Py_DECREF(entry);
    }
    return result;
}


static PyMethodDef PerTileQuality_methods[] = {
    {"add_read", (PyCFunction)PerTileQuality_add_read, 
     PerTileQuality_add_read_method, PerTileQuality_add_read__doc__},
    {"get_tile_averages", (PyCFunction)PerTileQuality_get_tile_averages,
     PerTileQuality_get_tile_averages_method, 
     PerTileQuality_get_tile_averages__doc__},
    {NULL},
};

static PyMemberDef PerTileQuality_members[] = {
    {"max_length", T_PYSSIZET, offsetof(PerTileQuality, max_length), READONLY, 
     "The length of the longest read"},
    {"number_of_reads", T_ULONGLONG, offsetof(PerTileQuality, number_of_reads), 
     READONLY, "The total amount of reads counted"},
    {"skipped_reason", T_OBJECT, offsetof(PerTileQuality, skipped_reason),
     READONLY, "What the reason is for skipping the module if skipped." 
               "Set to None if not skipped."},
    {NULL},
};

static PyTypeObject PerTileQuality_Type = {
    .tp_name = "_qc.PerTileQuality",
    .tp_basicsize = sizeof(PerTileQuality),
    .tp_dealloc = (destructor)PerTileQuality_dealloc,
    .tp_new = (newfunc)PerTileQuality__new__,
    .tp_members = PerTileQuality_members, 
    .tp_methods = PerTileQuality_methods,
};

/*************************
 * SEQUENCE DUPLICATION *
 *************************/

/* A module to use the first 50 bp of reads and collect the first 100_000 
   unique sequences and see how often they occur to estimate the duplication 
   rate and overrepresented sequences. The idea to take the first 100_000 with 
   50 bp comes from FastQC. 
   
   Below some typical figures:
   For a 5 million read illumina library:
    100000  Unique sequences recorded
     13038  Duplicates of one of the 100_000
   4886962  Not a duplicate of one of the 100_000

   A highly duplicated RNA library:
    100000  Unique sequences recorded
   1975437  Duplicates of one of the 100_000
   5856349  Not a duplicate of one of the 100_000
 
   As is visible, the most common case is that a read is not present in the
   first 100_000 unique sequences, even in the pathological case, so that case 
   should be optimized.
*/

#define MAX_UNIQUE_SEQUENCES 100000
#define UNIQUE_SEQUENCE_LENGTH 50
/* If size is a power of 2, the modulo HASH_TABLE_SIZE can be optimised to a
   bitwise AND by the compiler. Also this size (~262K entries) seems to work 
   well with a 100_000 slots in use. */
#define HASH_TABLE_SIZE (1ULL << 18)

/* This struct contains count, key_length and key on one single cache line 
   (64 bytes) so only one memory fetch is needed when a matching hash is found.*/
typedef struct _HashTableEntry {
    uint64_t count;
    uint8_t key_length;
    char key[55];  // 55 to align at 64 bytes. Only 50 is needed.
} HashTableEntry;

typedef struct _SequenceDuplicationStruct {
    PyObject_HEAD 
    size_t number_of_sequences;
    size_t number_of_uniques;
    Py_hash_t (*hashfunc)(const void *, Py_ssize_t);
    /* Store hashes and entries separately as in the most common case only
       the hash is needed. */
    Py_hash_t *hashes; 
    HashTableEntry *entries;
} SequenceDuplication;

static void 
SequenceDuplication_dealloc(SequenceDuplication *self)
{
    PyMem_Free(self->hashes);
    PyMem_Free(self->entries);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
SequenceDuplication__new__(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    static char *kwargnames[] = {NULL};
    static char *format = ":SequenceDuplication";
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, kwargnames)) {
        return NULL;
    }
    Py_hash_t *hashes = PyMem_Calloc(HASH_TABLE_SIZE, sizeof(Py_hash_t));
    HashTableEntry *entries = PyMem_Calloc(HASH_TABLE_SIZE, sizeof(HashTableEntry));
    PyHash_FuncDef *hashfuncdef = PyHash_GetFuncDef();
    if ((hashes == NULL) | (entries == NULL)) {
        PyMem_Free(hashes);
        PyMem_Free(entries);
        return PyErr_NoMemory();
    }
    SequenceDuplication *self = PyObject_New(SequenceDuplication, type);
    if (self == NULL) {
        return PyErr_NoMemory();
    }
    self->number_of_sequences = 0;
    self->number_of_uniques = 0;
    self->hashes = hashes;
    self->entries = entries;
    self->hashfunc = hashfuncdef->hash;
    return (PyObject *)self;
}


PyDoc_STRVAR(SequenceDuplication_add_read__doc__,
"add_read($self, read, /)\n"
"--\n"
"\n"
"Add a read to the duplication module. \n"
"\n"
"  read\n"
"    A FastqRecordView object.\n"
);

#define SequenceDuplication_add_read_method METH_O 

static PyObject *
SequenceDuplication_add_read(SequenceDuplication *self, FastqRecordView *read) 
{
    if (!FastqRecordView_CheckExact(read)) {
        PyErr_Format(PyExc_TypeError, 
                     "read should be a FastqRecordView object, got %s", 
                     Py_TYPE(read)->tp_name);
        return NULL;
    }
    self->number_of_sequences += 1;
    Py_ssize_t sequence_length = read->sequence_length;
    uint8_t *sequence = read->record_start + read->sequence_offset;
    Py_ssize_t hash_length = Py_MIN(sequence_length, UNIQUE_SEQUENCE_LENGTH);
    Py_hash_t hash = self->hashfunc(sequence, hash_length);
    /* Ensure hash is never 0, because that is reserved for empty slots. By 
       setting the most significant bit, this does not affect the resulting index. */
    hash |= (1ULL << 63);  
    Py_hash_t *hashes = self->hashes;
    size_t index = hash % HASH_TABLE_SIZE;
 
    while (1) {
        Py_hash_t hash_entry = hashes[index];
        if (hash_entry == 0) {
            if (self->number_of_uniques < MAX_UNIQUE_SEQUENCES) {
                hashes[index] = hash;
                HashTableEntry *entry = self->entries + index;
                entry->count = 1;
                entry->key_length = hash_length;
                memcpy(entry->key, sequence, hash_length);
                self->number_of_uniques += 1;
            }
            break;
        } else if (hash_entry == hash) {
                HashTableEntry *entry = self->entries + index;
            /* There is a very small chance of a hash collision, check to make 
               sure. If not equal we simply go to the next hash_entry. */
            if (entry->key_length == hash_length && 
                memcmp(entry->key, sequence, hash_length) == 0) {
                entry->count += 1;
                break;
            }
        }
        index += 1;
        /* Make sure the index round trips when it reaches HASH_TABLE_SIZE.*/
        index %= HASH_TABLE_SIZE;
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(SequenceDuplication_sequence_counts__doc__,
"sequence_counts($self, /)\n"
"--\n"
"\n"
"Get a dictionary with sequence counts \n"
);

#define SequenceDuplication_sequence_counts_method METH_NOARGS

static PyObject *
SequenceDuplication_sequence_counts(SequenceDuplication *self, PyObject *Py_UNUSED(ignore))
{
    PyObject *count_dict = PyDict_New();
    if (count_dict == NULL) {
        return PyErr_NoMemory();
    }
    HashTableEntry *entries = self->entries;

    for (size_t i=0; i < HASH_TABLE_SIZE; i+=1) {
        HashTableEntry *entry = entries + i;
        if (entry->count == 0) {
            continue;
        }
        PyObject *count_obj = PyLong_FromUnsignedLongLong(entry->count);
        if (count_obj == NULL) {
            goto error;
        }
        PyObject *key = PyUnicode_New(entry->key_length, 127);
        if (key == NULL) {
            goto error;
        }
        memcpy(PyUnicode_DATA(key), entry->key, entry->key_length);
        if (PyDict_SetItem(count_dict, key, count_obj) != 0) {
            goto error;
        }
        Py_DECREF(count_obj);
        Py_DECREF(key);
    }
    return count_dict;

error:
    Py_DECREF(count_dict);
    return NULL;
}

PyDoc_STRVAR(SequenceDuplication_overrepresented_sequences__doc__,
"overrepresented_sequences($self, threshold=0.001)\n"
"--\n"
"\n"
"Return a list of tuples with the fraction and the sequence. The list is "
"sorted in reverse order with the most common sequence on top.\n"
"\n"
"  threshold\n"
"    The fraction at which a sequence is considered overrepresented.\n"
);

#define SequenceDuplication_overrepresented_sequences_method METH_VARARGS | METH_KEYWORDS

static PyObject *
SequenceDuplication_overrepresented_sequences(SequenceDuplication *self, 
                                              PyObject *args, PyObject *kwargs)
{
    double threshold = 0.001;  // 0.1 %
    static char *kwargnames[] = {"threshold", NULL};
    static char *format = "|d:SequenceDuplication.overrepresented_sequences";
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, kwargnames, &threshold)) {
        return NULL;
    }
    if ((threshold < 0.0) || (threshold > 1.0)) {
        // PyErr_Format has no direct way to represent floats
        PyObject *threshold_obj = PyFloat_FromDouble(threshold);
        PyErr_Format(
            PyExc_ValueError, 
            "threshold must be between 0.0 and 1.0 got, %R", threshold_obj, 
            threshold);
        Py_XDECREF(threshold_obj);
        return NULL;
    }

    PyObject *result = PyList_New(0);
    if (result == NULL) {
        return NULL;
    }

    uint64_t total_sequences = self->number_of_sequences;
    uint64_t minimum_hits = threshold * total_sequences;
    HashTableEntry *entries = self->entries;

    for (size_t i=0; i < HASH_TABLE_SIZE; i+=1) {
        HashTableEntry *entry = entries + i;
        uint64_t count = entry->count;
        if (count == 0) {
            continue;
        }
        if (count >= minimum_hits) {
            PyObject *entry_tuple = Py_BuildValue(
                "(ds#)", 
                ((double)count / (double)total_sequences),
                 entry->key, entry->key_length);
            if (entry_tuple == NULL) {
                goto error;
            }
            if (PyList_Append(result, entry_tuple) != 0) {
                goto error;
            }
            Py_DECREF(entry_tuple);
        }
    }
    /* Sort and reverse the list so the most common entries are at the top */
    if (PyList_Sort(result) != 0) {
        goto error;
    }
    if (PyList_Reverse(result) != 0) {
        goto error;
    }
    return result;
error: 
    Py_DECREF(result);
    return NULL;
}

PyDoc_STRVAR(SequenceDuplication_duplication_counts__doc__,
"duplication_counts($self, max_count=50_000)\n"
"--\n"
"\n"
"Return a count_array of values such that count_array[1] returns the count "
"of sequences that were only seen once, count_array[5:10] returns those that "
"were seen 5-9 times. count_array[max_count] gives the number of "
"sequences that were equal or more than max_count.\n"
"\n"
"  threshold\n"
"    The fraction at which a sequence is considered overrepresented.\n"
);

#define SequenceDuplication_duplication_counts_method METH_VARARGS | METH_KEYWORDS

static PyObject *
SequenceDuplication_duplication_counts(SequenceDuplication *self, 
                                       PyObject *args, PyObject *kwargs)
{
    static char *kwargnames[] = {"max_count", NULL};
    static char *format = "|n:SequenceDuplication.duplication_counts";
    Py_ssize_t max_count_signed = 50000;
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, kwargnames, 
            &max_count_signed)) {
        return NULL;
    }
    if (max_count_signed < 1) {
        PyErr_Format(PyExc_ValueError, 
                     "Max count needs to be at least one, got %z", 
                     max_count_signed);
        return NULL;
    }
    size_t max_count = max_count_signed;
    uint64_t *count_array = PyMem_Calloc(max_count + 1, sizeof(uint64_t));
    if (count_array == NULL) {
        return PyErr_NoMemory();
    }
    HashTableEntry *entries = self->entries;

    for (size_t i=0; i < HASH_TABLE_SIZE; i+=1) {
        HashTableEntry *entry = entries + i;
        uint64_t count = entry->count;
        if (count == 0) {
            continue;
        }
        if (count > max_count) {
            count = max_count;
        }
        count_array[count] += 1;
    }
    PyObject *result = PythonArray_FromBuffer('Q', count_array, (max_count + 1) * sizeof(uint64_t));
    PyMem_Free(count_array);
    return result;
}

static PyMethodDef SequenceDuplication_methods[] = {
    {"add_read", (PyCFunction)SequenceDuplication_add_read, 
     SequenceDuplication_add_read_method, 
     SequenceDuplication_add_read__doc__},
    {"sequence_counts", (PyCFunction)SequenceDuplication_sequence_counts,
     SequenceDuplication_sequence_counts_method, 
     SequenceDuplication_sequence_counts__doc__},
    {"overrepresented_sequences", 
     (PyCFunction)(void(*)(void))SequenceDuplication_overrepresented_sequences,
      SequenceDuplication_overrepresented_sequences_method,
      SequenceDuplication_overrepresented_sequences__doc__},
    {"duplication_counts", 
     (PyCFunction)(void(*)(void))SequenceDuplication_duplication_counts,
     SequenceDuplication_duplication_counts_method,
     SequenceDuplication_duplication_counts__doc__},
    {NULL},
};

static PyMemberDef SequenceDuplication_members[] = {
    {"number_of_sequences", T_ULONGLONG, 
     offsetof(SequenceDuplication, number_of_sequences), READONLY,
     "The total number of sequences processed"},
    {NULL},
};

static PyTypeObject SequenceDuplication_Type = {
    .tp_name = "_qc.SequenceDuplication",
    .tp_basicsize = sizeof(SequenceDuplication),
    .tp_dealloc = (destructor)(SequenceDuplication_dealloc),
    .tp_new = (newfunc)SequenceDuplication__new__,
    .tp_members = SequenceDuplication_members,
    .tp_methods = SequenceDuplication_methods,
};


/*************************
 * MODULE INITIALIZATION *
 *************************/


static struct PyModuleDef _qc_module = {
    PyModuleDef_HEAD_INIT,
    "_qc",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,
    NULL,  /* module methods */
    .m_slots = NULL,
};

/* A C implementation of from module_name import class_name*/
static PyTypeObject *ImportClassFromModule(
    const char *module_name, 
    const char *class_name) 
{
    PyObject *module = PyImport_ImportModule(module_name);
    if (module == NULL) {
        return NULL;
    }
    PyTypeObject *type_object = (PyTypeObject *)PyObject_GetAttrString(
        module, class_name);
    if (type_object == NULL) {
        return NULL;
    }
    if (!PyType_CheckExact(type_object)) {
        PyErr_Format(PyExc_RuntimeError, 
            "%s.%s is not a type class but, %s",
            module_name, class_name, Py_TYPE(type_object)->tp_name);
        return NULL;
    }
    return type_object;
}

/* Simple reimplementation of PyModule_AddType given that this is only available
   from python 3.9 onwards*/
static int 
python_module_add_type(PyObject *module, PyTypeObject *type)
{
    if (PyType_Ready(type) != 0) {
        return -1;
    }
    const char *class_name = strchr(type->tp_name, '.');
    if (class_name == NULL) {
        return -1;
    }
    class_name += 1; // Use the part after the dot.
    Py_INCREF(type);
    if (PyModule_AddObject(module, class_name, (PyObject *)type) != 0) {
        return -1;
    }
    return 0;
}

PyMODINIT_FUNC
PyInit__qc(void)
{
    PyObject *m = PyModule_Create(&_qc_module);
    if (m == NULL) {
        return NULL;
    }

    PythonArray = ImportClassFromModule("array", "array");
    SequenceRecord = ImportClassFromModule("dnaio", "SequenceRecord");
    if ((PythonArray == NULL) || (SequenceRecord == NULL)) {
        return NULL;
    }
    if (python_module_add_type(m, &FastqParser_Type) != 0) {
        return NULL;
    }  
    if (python_module_add_type(m, &FastqRecordView_Type) != 0) {
        return NULL; 
    }
    if (python_module_add_type(m, &QCMetrics_Type) != 0) {
        return NULL;
    }
    if (python_module_add_type(m, &AdapterCounter_Type) != 0) {
        return NULL;
    }
    if (python_module_add_type(m, &PerTileQuality_Type) != 0) {
        return NULL;
    }
    if (python_module_add_type(m, &SequenceDuplication_Type) != 0) {
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
    PyModule_AddIntMacro(m, MAX_SEQUENCE_SIZE);
    return m;
}
