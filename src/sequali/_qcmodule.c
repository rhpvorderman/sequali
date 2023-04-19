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


/**************
 * QC METRICS *
 **************/

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

static PyMethodDef QCMetrics_methods[] = {
    QCMETRICS_ADD_READ_METHODDEF,
    QCMETRICS_COUNT_TABLE_VIEW_METHODDEF,
    QCMETRICS_GC_CONTENT_VIEW_METHODDEF,
    QCMETRICS_PHRED_SCORES_VIEW_METHODDEF,
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

PyDoc_STRVAR(AdapterCounter_add_sequence__doc__,
"add_sequence($self, sequence, /)\n"
"--\n"
"\n"
"Add a sequence to the adapter counter. \n"
"\n"
"  sequence\n"
"    An ASCII string containing the sequence.\n"
);

#define ADAPTERCOUNTER_ADD_SEQUENCE_METHODDEF    \
    {"add_sequence", (PyCFunction)(void(*)(void))AdapterCounter_add_sequence, \
    METH_O, AdapterCounter_add_sequence__doc__}

static PyObject *
AdapterCounter_add_sequence(AdapterCounter *self, PyObject *sequence_obj) 
{
    if (!PyUnicode_CheckExact(sequence_obj)) {
        PyErr_Format(PyExc_TypeError, "sequence should be a str, got %s", 
                     Py_TYPE(sequence_obj)->tp_name);
        return NULL;
    }
    if (!PyUnicode_IS_COMPACT_ASCII(sequence_obj)) {
        PyErr_Format(PyExc_ValueError, 
                     "Sequence should only contain ASCII characters: %R",
                     sequence_obj);
        return NULL;
    }
    self->number_of_sequences += 1;
    uint8_t *sequence = PyUnicode_DATA(sequence_obj);
    size_t sequence_length = PyUnicode_GET_LENGTH(sequence_obj);

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
"and a memoryview to the counts per position. \n"
);

#define ADAPTERCOUNTER_GET_COUNTS_METHODDEF    \
    {"get_counts", (PyCFunction)(void(*)(void))AdapterCounter_get_counts, \
    METH_NOARGS, AdapterCounter_get_counts__doc__}

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
        Py_buffer buf = {
            .buf = self->adapter_counter[i],
            .obj = NULL,
            .len = self->max_length * sizeof(counter_t),
            .readonly = 1,
            .itemsize = sizeof(counter_t),
            .format = "Q",
            .ndim = 1,
        };
        PyObject *view = PyMemoryView_FromBuffer(&buf);
        if (view == NULL) {
            return NULL;
        }
        PyObject *adapter = PyTuple_GET_ITEM(self->adapters, i);
        Py_INCREF(adapter);
        PyTuple_SET_ITEM(tup, 0, adapter);
        PyTuple_SET_ITEM(tup, 1, view);
        PyList_SET_ITEM(counts_list, i, tup);
    }
    return counts_list;
}


static PyMethodDef AdapterCounter_methods[] = {
    ADAPTERCOUNTER_ADD_SEQUENCE_METHODDEF,
    ADAPTERCOUNTER_GET_COUNTS_METHODDEF,
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
    PyObject *header_name;
    PyObject *qual_name;
    uint8_t phred_offset;
    char skipped;
    BaseQuality **base_qualities;
    size_t number_of_tiles;
    Py_ssize_t max_length;
    size_t number_of_reads;
    PyObject *skipped_reason;
} PerTileQuality;

static void
PerTileQuality_dealloc(PerTileQuality *self) {
    Py_DECREF(self->header_name);
    Py_DECREF(self->qual_name);
    Py_XDECREF(self->skipped_reason);
    for (size_t i=0; i < self->number_of_tiles; i++) {
        BaseQuality *tile_quals = self->base_qualities[i];
        PyMem_Free(tile_quals);
    }
    PyMem_Free(self->base_qualities);
}

static PyObject *
PerTileQuality__new__(PyTypeObject *type, PyObject *args, PyObject *kwargs){
    static char *kwargnames[] = {NULL};
    static char *format = ":PerTileQuality";
    uint8_t phred_offset = 33;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, kwargnames)) {
        return NULL;
    }
    PyObject *header_name = PyUnicode_FromString("name");
    if (header_name == NULL) {
        return NULL;
    }
    PyObject *qual_name = PyUnicode_FromString("qualities");
    if (qual_name == NULL) {
        return NULL;
    }

    PerTileQuality *self = PyObject_New(PerTileQuality, type);
    self->max_length = 0;
    self->phred_offset = phred_offset;
    self->base_qualities = NULL;
    self->number_of_reads = 0;
    self->header_name = header_name;
    self->qual_name = qual_name;
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

#define PERTILEQUALITY_ADD_READ_METHODDEF    \
    {"add_read", (PyCFunction)(void(*)(void))PerTileQuality_add_read, METH_O, \
     PerTileQuality_add_read__doc__}

static PyObject *
PerTileQuality_add_read(PerTileQuality *self, PyObject *read)
{
    if (self->skipped) {
        Py_RETURN_NONE;
    }
    if (Py_TYPE(read) != SequenceRecord) {
        PyErr_Format(PyExc_TypeError,
                     "Read should be a dnaio.SequenceRecord object, got %s",
                     Py_TYPE(read)->tp_name);
        return NULL;
    }
    /* PyObject_GetAttrString creates a new unicode object every single time.
       Use PyObject_GetAttr to prevent this. */
    PyObject *header_obj = PyObject_GetAttr(read, self->header_name);
    PyObject *qualities_obj = PyObject_GetAttr(read, self->qual_name);
    /* Dnaio guarantees ASCII strings */
    const char *header = PyUnicode_DATA(header_obj);
    Py_ssize_t header_length = PyUnicode_GET_LENGTH(header_obj);
    const uint8_t *qualities = PyUnicode_DATA(qualities_obj);
    Py_ssize_t sequence_length = PyUnicode_GET_LENGTH(qualities_obj);
    size_t i;
    uint8_t phred_offset = self->phred_offset;
    uint8_t q;

    long tile_id = illumina_header_to_tile_id(header, header_length);
    if (tile_id == -1) {
        self->skipped_reason = PyUnicode_FromFormat(
            "Can not parse header: %s", (char *)header); 
        self->skipped = 1;
        goto success;
    }

    if (sequence_length > self->max_length) {
        if (PerTileQuality_resize_tiles(self, sequence_length) != 0) {
            goto error;
        }
    }

    /* Tile index must be one less than the highest number of tiles otherwise 
       the index is not in the tile array. */
    if (((size_t)tile_id + 1) > self->number_of_tiles) {
        if (PerTileQuality_resize_tile_array(self, tile_id + 1) != 0) {
            goto error;
        }
    }
    
    BaseQuality *tile_qualities = self->base_qualities[tile_id];
    if (tile_qualities == NULL) {
        tile_qualities = PyMem_Malloc(self->max_length * sizeof(BaseQuality));
        if (tile_qualities == NULL) {
            PyErr_NoMemory();
            goto error;
        }
        memset(tile_qualities, 0, self->max_length * sizeof(BaseQuality));
        self->base_qualities[tile_id] = tile_qualities;
    }

    self->number_of_reads += 1;
    for (i=0; i < (size_t)sequence_length; i+=1) {
        q = qualities[i] - phred_offset;
        if (q > PHRED_MAX) {
            PyErr_Format(
                PyExc_ValueError,
                "Not a valid phred character: %c", qualities[i]
            );
            goto error;
        }
        tile_qualities[i].total_bases += 1;
        tile_qualities[i].total_error += SCORE_TO_ERROR_RATE[q];
    }

success:
    Py_DECREF(header_obj);
    Py_DECREF(qualities_obj);
    Py_RETURN_NONE;
error:
    Py_DECREF(header_obj);
    Py_DECREF(qualities_obj);
    return NULL;
}


PyDoc_STRVAR(PerTileQuality_get_tile_averages__doc__,
"add_read($self, /)\n"
"--\n"
"\n"
"Get a list of tuples with the tile IDs and a list of their averages. \n"
);

#define PERTILEQUALITY_GET_TILE_AVERAGES_METHODDEF    \
    {"get_tile_averages", (PyCFunction)(void(*)(void))PerTileQuality_get_tile_averages, \
    METH_NOARGS, PerTileQuality_get_tile_averages__doc__}

static PyObject *
PerTileQuality_get_tile_averages(PerTileQuality *self, PyObject *read)
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
    PERTILEQUALITY_ADD_READ_METHODDEF,
    PERTILEQUALITY_GET_TILE_AVERAGES_METHODDEF,
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
   unique sequences and see how often
   they occur to estimate the duplication rate and overrepresented sequences. 
   The idea to take the first 100_000 with 50 bp comes from FastQC. Below 
   some typical figures:
   For a 5 million read illumina library:
   13038 Found before
   100000 Inserted
   4886962 Not present
   
   100_000 Unique sequences were added. 13.038 sequences belonged to one of 
   these 100_000. The vast majority of the sequences did not belong to the first
   100_000.

   A highly duplicated RNA library:
   1975437 Found before
   100000 Inserted
   5856349 Not present
 
   Almost 2 million sequences were found that were identical tot the first 
   100_000 unique reads. 

   As is visible, the most common case is that a read is not present in the
   first 100_000 unique sequences, even in the pathological case, so that case 
   should be optimized.
*/

#define MAX_UNIQUE_SEQUENCES 100000
#define UNIQUE_SEQUENCE_LENGTH 50
#define HASH_TABLE_SIZE (1ULL << 18)

static inline size_t hash_to_index(Py_hash_t hash) {
    /* No modulo required because HASH_TABLE_SIZE is a power of 2 */
    return hash & (HASH_TABLE_SIZE - 1);
}

typedef struct _HashTableEntry {
    uint64_t count;
    uint8_t key_length;
    char key[55];  // 55 to align at 64 bytes. Only 50 is needed.
} HashTableEntry;

typedef struct _SequenceDuplicationStruct {
    PyObject_HEAD 
    size_t number_of_sequences;
    size_t number_of_uniques;
    Py_hash_t *hash_table; 
    Py_hash_t (*hashfunc)(const void *, Py_ssize_t);
    HashTableEntry *entries;
} SequenceDuplication;

static void 
SequenceDuplication_dealloc(SequenceDuplication *self)
{
    PyMem_Free(self->hash_table);
    PyMem_Free(self->entries);
}

static PyObject *
SequenceDuplication__new__(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    static char *kwargnames[] = {NULL};
    static char *format = ":SequenceDuplication";
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, kwargnames)) {
        return NULL;
    }
    Py_hash_t *hash_table = PyMem_Calloc(HASH_TABLE_SIZE, sizeof(Py_hash_t));
    HashTableEntry *entries = PyMem_Calloc(HASH_TABLE_SIZE, sizeof(HashTableEntry));
    PyHash_FuncDef *hashfuncdef = PyHash_GetFuncDef();
    if ((hash_table == NULL) | (entries == NULL)) {
        PyMem_Free(hash_table);
        PyMem_Free(entries);
        return PyErr_NoMemory();
    }
    SequenceDuplication *self = PyObject_New(SequenceDuplication, type);
    if (self == NULL) {
        return PyErr_NoMemory();
    }
    self->number_of_sequences = 0;
    self->number_of_uniques = 0;
    self->hash_table = hash_table;
    self->entries = entries;
    self->hashfunc = hashfuncdef->hash;
    return (PyObject *)self;
}


PyDoc_STRVAR(SequenceDuplication_add_sequence__doc__,
"add_sequence($self, sequence, /)\n"
"--\n"
"\n"
"Add a sequence to the duplication module. \n"
"\n"
"  sequence\n"
"    An ASCII string containing the sequence.\n"
);

#define SEQUENCEDUPLICATION_ADD_SEQUENCE_METHODDEF    \
    {"add_sequence", (PyCFunction)(void(*)(void))SequenceDuplication_add_sequence, \
    METH_O, SequenceDuplication_add_sequence__doc__}

static PyObject *
SequenceDuplication_add_sequence(SequenceDuplication *self, PyObject *sequence_obj) 
{
    if (!PyUnicode_CheckExact(sequence_obj)) {
        PyErr_Format(PyExc_TypeError, "sequence should be a str, got %s", 
                     Py_TYPE(sequence_obj)->tp_name);
        return NULL;
    }
    if (!PyUnicode_IS_COMPACT_ASCII(sequence_obj)) {
        PyErr_Format(PyExc_ValueError, 
                     "Sequence should only contain ASCII characters: %R",
                     sequence_obj);
        return NULL;
    }
    self->number_of_sequences += 1;
    Py_ssize_t sequence_length = PyUnicode_GET_LENGTH(sequence_obj);
    char *sequence = PyUnicode_DATA(sequence_obj);
    Py_ssize_t hash_length = Py_MIN(sequence_length, UNIQUE_SEQUENCE_LENGTH);
    Py_hash_t hash = self->hashfunc(sequence, hash_length);
    Py_hash_t *hash_table = self->hash_table;
    size_t index = hash_to_index(hash);
 
    while (1) {
        Py_hash_t hash_entry = hash_table[index];
        if (hash_entry == 0) {
            if (self->number_of_uniques < MAX_UNIQUE_SEQUENCES) {
                hash_table[index] = hash;
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
        /* Make sure the index round trips when it reaches HASH_TABLE_SIZE. 
           The &= works for hash table sizes that are a power of 2. */
        index &= (HASH_TABLE_SIZE - 1);
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(SequenceDuplication_sequence_counts__doc__,
"sequence_counts($self, /)\n"
"--\n"
"\n"
"Get a dictionary with sequence counts \n"
);

#define SEQUENCEDUPLICATION_SEQUENCE_COUNTS_METHODDEF    \
    {"sequence_counts", (PyCFunction)(void(*)(void))SequenceDuplication_sequence_counts, \
    METH_NOARGS, SequenceDuplication_sequence_counts__doc__}

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


static PyMethodDef SequenceDuplication_methods[] = {
    SEQUENCEDUPLICATION_ADD_SEQUENCE_METHODDEF,
    SEQUENCEDUPLICATION_SEQUENCE_COUNTS_METHODDEF,
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

    if (PyType_Ready(&AdapterCounter_Type) != 0) {
        return NULL;
    }
    Py_INCREF(&AdapterCounter_Type);
    if (PyModule_AddObject(m, "AdapterCounter", 
                           (PyObject *)&AdapterCounter_Type) != 0) {
        return NULL;
    }
    
    if (PyType_Ready(&PerTileQuality_Type) != 0) {
        return NULL;
    }
    Py_INCREF(&PerTileQuality_Type);
    if (PyModule_AddObject(m, "PerTileQuality", 
                           (PyObject *)&PerTileQuality_Type) != 0) {
        return NULL;
    }

    if (PyType_Ready(&SequenceDuplication_Type) != 0) {
        return NULL;
    } 
    Py_INCREF(&SequenceDuplication_Type);
    if (PyModule_AddObject(m, "SequenceDuplication",
                          (PyObject *)&SequenceDuplication_Type) != 0) {
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
