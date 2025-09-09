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

#define Py_LIMITED_API 0x030A0000
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "structmember.h"

/* Buffer constants have remained the same and it is a shame not to be
   available for python versions below 3.11 for just these two missing
   constants. */
#ifndef PyBUF_READ
#define PyBUF_READ 0x100
#define PyBUF_WRITE 0x200
#endif

#include "compiler_defs.h"
#include "murmur3.h"
#include "score_to_error_rate.h"
#include "wanghash.h"

#include <math.h>
#include <stdbool.h>

/* Pointers to types that will be imported/initialized in the module
   initialization section */

struct QCModuleState {
    PyTypeObject *PythonArray_Type;  // array.array
    PyTypeObject *FastqRecordView_Type;
    PyTypeObject *FastqRecordArrayView_Type;
    PyTypeObject *FastqParser_Type;
    PyTypeObject *BamParser_Type;
    PyTypeObject *QCMetrics_Type;
    PyTypeObject *AdapterCounter_Type;
    PyTypeObject *OverrepresentedSequences_Type;
    PyTypeObject *DedupEstimator_Type;
    PyTypeObject *PerTileQuality_Type;
    PyTypeObject *NanoporeReadInfo_Type;
    PyTypeObject *NanoStats_Type;
    PyTypeObject *NanoStatsIterator_Type;
    PyTypeObject *InsertSizeMetrics_Type;
};

static inline struct QCModuleState *
get_qc_module_state_from_type(PyTypeObject *tp)
{
    return (struct QCModuleState *)PyType_GetModuleState(tp);
}

/**
 * @brief Get the qc module state from obj object. Object is a void to allow
 *        custom classes to be passed in.
 *
 * @param obj
 * @return struct QCModuleState*
 */
static inline struct QCModuleState *
get_qc_module_state_from_obj(void *obj)
{
    return get_qc_module_state_from_type(Py_TYPE((PyObject *)obj));
}

static inline int
is_FastqRecordView(void *module_obj, void *obj_to_check)
{
    struct QCModuleState *state = get_qc_module_state_from_obj(module_obj);
    if (state == NULL) {
        return -1;
    }
    return PyObject_IsInstance(obj_to_check,
                               (PyObject *)state->FastqRecordView_Type);
}

static inline int
is_FastqRecordArrayView(void *module_obj, void *obj_to_check)
{
    struct QCModuleState *state = get_qc_module_state_from_obj(module_obj);
    if (state == NULL) {
        return -1;
    }
    return PyObject_IsInstance(obj_to_check,
                               (PyObject *)state->FastqRecordArrayView_Type);
}

#define PHRED_MAX 93

/*********
 * Utils *
 *********/

static inline void
non_temporal_write_prefetch(void *address)
{
#if __GNUC__ || CLANG_COMPILER_HAS_BUILTIN(__builtin_prefetch)
    __builtin_prefetch(address, 1, 0);
#elif BUILD_IS_X86_64
    /* Fallback for known architecture */
    _mm_prefetch(address, _MM_HINT_NTA);
#else
/* No-op for MSVC and other compilers. MSVC builtin was not found. */
#endif
}

static PyObject *
PythonArray_FromBuffer(char typecode, void *buffer, size_t buffersize,
                       PyTypeObject *PythonArray_Type)
{
    PyObject *array =
        PyObject_CallFunction((PyObject *)PythonArray_Type, "C", typecode);
    if (array == NULL) {
        return NULL;
    }
    if (buffersize == 0) {
        /* Return empty array */
        return array;
    }
    /* We cannot paste into the array directly, so use a temporary memoryview */
    PyObject *tmp = PyMemoryView_FromMemory(buffer, buffersize, PyBUF_READ);
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

/**
 * @brief Simple strtoul replacement.
 *
 * Can be inlined easily by the compiler, as well as quick loop unrolling and
 * removing checks if the number of digits is given.
 *
 * @param string The string pointing to an unsigned decimal number
 * @param length The length of the number string
 * @return Py_ssize_t the answer, or -1 on error.
 */
static inline Py_ssize_t
unsigned_decimal_integer_from_string(const uint8_t *string, size_t length)
{
    /* There should be at least one digit and larger than 18 digits can not
       be stored due to overflow */
    if (length < 1 || length > 18) {
        return -1;
    }
    size_t result = 0;
    for (size_t i = 0; i < length; i++) {
        uint8_t c = string[i];
        /* 0-9 range check. Only one side needs to be checked because of
            unsigned number */
        c -= '0';
        if (c > 9) {
            return -1;
        }
        /* Shift already found digits one decimal place and add current digit */
        result = result * 10 + c;
    }
    return result;
}

#define ASCII_MASK_8BYTE 0x8080808080808080ULL
#define ASCII_MASK_1BYTE 0x80

static inline int
string_is_ascii_fallback(const char *string, size_t length)
{
    size_t all_chars = 0;
    for (size_t i = 0; i < length; i++) {
        all_chars |= string[i];
    }
    return !(all_chars & ASCII_MASK_1BYTE);
}

/**
 * @brief Check if a string of given length only contains ASCII characters.
 *
 * @param string A char pointer to the start of the string.
 * @param length The length of the string. This funtion does not check for
 *               terminating NULL bytes.
 * @returns 1 if the string is ASCII-only, 0 otherwise.
 */
static int
string_is_ascii(const char *string, size_t length)
{
    if (length < sizeof(size_t)) {
        return string_is_ascii_fallback(string, length);
    }
    size_t number_of_chunks = length / sizeof(size_t);
    size_t *chunks = (size_t *)string;
    size_t number_of_unrolls = number_of_chunks / 4;
    size_t remaining_chunks = number_of_chunks - (number_of_unrolls * 4);
    size_t *chunk_ptr = chunks;
    size_t all_chars0 = 0;
    size_t all_chars1 = 0;
    size_t all_chars2 = 0;
    size_t all_chars3 = 0;
    for (size_t i = 0; i < number_of_unrolls; i++) {
        /* Performing indepedent OR calculations allows the compiler to use
           vectors. It also allows out of order execution. */
        all_chars0 |= chunk_ptr[0];
        all_chars1 |= chunk_ptr[1];
        all_chars2 |= chunk_ptr[2];
        all_chars3 |= chunk_ptr[3];
        chunk_ptr += 4;
    }
    size_t all_chars = all_chars0 | all_chars1 | all_chars2 | all_chars3;
    for (size_t i = 0; i < remaining_chunks; i++) {
        all_chars |= chunk_ptr[i];
    }
    /* Load the last few bytes left in a single integer for fast operations.
       There is some overlap here with the work done before, but for a simple
       ascii check this does not matter. */
    size_t last_chunk = *(size_t *)(string + length - sizeof(size_t));
    all_chars |= last_chunk;
    return !(all_chars & ASCII_MASK_8BYTE);
}

/***
 * Seconds since epoch for years of 1970 and higher is defined in the POSIX
 * specification.:
 * https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap04.html#tag_04_16
 * With a little help from Eric S. Raymond's solution in this stackoverflow
 * answer:
 * https://stackoverflow.com/questions/530519/stdmktime-and-timezone-info
 */
static inline time_t
posix_gm_time(time_t year, time_t month, time_t mday, time_t hour,
              time_t minute, time_t second)
{
    /* Following code is only true for years equal or greater than 1970*/
    if (year < 1970 || month < 1 || month > 12) {
        return -1;
    }
    year -= 1900;  // Years are relative to 1900
    static const int mday_to_yday[12] = {0,   31,  59,  90,  120, 151,
                                         181, 212, 243, 273, 304, 334};
    time_t yday = mday_to_yday[month - 1] + mday - 1;
    return second + minute * 60 + hour * 3600 + yday * 86400 +
           (year - 70) * 31536000 + ((year - 69) / 4) * 86400 -
           ((year - 1) / 100) * 86400 + ((year + 299) / 400) * 86400;
}

/**
 * @brief Convert a timestring in Nanopore format to a timestamp.
 *
 * @param time_string A string in year-month-dateThour:minute:secondZ format.
 * @return time_t The unix timestamp, -1 on failure. Nanopore was not invented
 *          on New Year's eve 1969 so this should not lead to confusion ;-).
 */
static time_t
time_string_to_timestamp(const uint8_t *time_string)
{
    /* Time format used 2019-01-26T18:52:46Z
       Could be parsed with sscanf, but it is much quicker to completely inline
       the call by using an inlinable function. */
    const uint8_t *s = time_string;
    Py_ssize_t year = unsigned_decimal_integer_from_string(s, 4);
    Py_ssize_t month = unsigned_decimal_integer_from_string(s + 5, 2);
    Py_ssize_t day = unsigned_decimal_integer_from_string(s + 8, 2);
    Py_ssize_t hour = unsigned_decimal_integer_from_string(s + 11, 2);
    Py_ssize_t minute = unsigned_decimal_integer_from_string(s + 14, 2);
    Py_ssize_t second = unsigned_decimal_integer_from_string(s + 17, 2);
    /* If one of year, month etc. is -1 the signed bit is set. Bitwise OR
       allows checking them all at once for this. */
    if ((year | month | day | hour | minute | second) < 0 || s[4] != '-' ||
        s[7] != '-' || s[10] != 'T' || s[13] != ':' || s[16] != ':') {
        return -1;
    }
    const uint8_t *tz_part = s + 19;
    /* Sometimes there is a miliseconds parts that needs to be parsed */
    if (*tz_part == '.') {
        size_t decimal_size = strspn((char *)s + 20, "0123456789");
        tz_part += decimal_size + 1;
    }
    Py_ssize_t offset_hours;
    Py_ssize_t offset_minutes;
    switch (tz_part[0]) {
        case 'Z':
            /* UTC No special code needed. */
            break;
        case '+':
        case '-':
            offset_hours = unsigned_decimal_integer_from_string(tz_part + 1, 2);
            offset_minutes = unsigned_decimal_integer_from_string(tz_part + 4, 2);
            if ((offset_hours | offset_minutes) < 0 || tz_part[3] != ':') {
                return -1;
            }
            if ((tz_part[0]) == '+') {
                hour += offset_hours;
                minute += offset_minutes;
            }
            else {
                hour -= offset_hours;
                minute -= offset_minutes;
            }
            break;
        default:
            return -1;
    }
    return posix_gm_time(year, month, day, hour, minute, second);
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

struct FastqMeta {
    union {
        uint8_t *record_start;
        uint8_t *name;  // Name is always at the start of the record.
    };
    uint32_t name_length;
    uint32_t sequence_offset;
    // Sequence length and qualities length should be the same
    union {
        uint32_t sequence_length;
        uint32_t qualities_length;
    };
    uint32_t qualities_offset;
    uint32_t tags_offset;
    uint32_t tags_length;
    /* Store the accumulated error once calculated so it can be reused by
       the NanoStats module */
    double accumulated_error_rate;
};

typedef struct _FastqRecordViewStruct {
    PyObject_HEAD
    struct FastqMeta meta;
    PyObject *obj;
} FastqRecordView;

static void
FastqRecordView_dealloc(FastqRecordView *self)
{
    Py_XDECREF(self->obj);
    PyObject *tp = (PyObject *)Py_TYPE((PyObject *)self);
    PyObject_Free(self);
    Py_DECREF(tp);
}

static PyObject *
FastqRecordView__new__(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyObject *name_obj = NULL;
    PyObject *sequence_obj = NULL;
    PyObject *qualities_obj = NULL;
    PyObject *tags_obj = NULL;
    static char *kwargnames[] = {"name", "sequence", "qualities", "tags", NULL};
    static char *format = "UUU|S:FastqRecordView";
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, kwargnames, &name_obj,
                                     &sequence_obj, &qualities_obj, &tags_obj)) {
        return NULL;
    }
    // Unicode check already performed by the parser, so error checking can be
    // ignored in some cases.
    Py_ssize_t original_name_length = PyUnicode_GetLength(name_obj);
    Py_ssize_t name_length = 0;
    const uint8_t *name =
        (const uint8_t *)PyUnicode_AsUTF8AndSize(name_obj, &name_length);
    if (original_name_length != name_length) {
        PyErr_Format(PyExc_ValueError,
                     "name should contain only ASCII characters: %R", name_obj);
        return NULL;
    }

    Py_ssize_t original_sequence_length = PyUnicode_GetLength(sequence_obj);
    Py_ssize_t sequence_length = 0;
    const uint8_t *sequence =
        (const uint8_t *)PyUnicode_AsUTF8AndSize(sequence_obj, &sequence_length);
    if (original_sequence_length != sequence_length) {
        PyErr_Format(PyExc_ValueError,
                     "sequence should contain only ASCII characters: %R",
                     sequence_obj);
        return NULL;
    }

    Py_ssize_t original_qualities_length = PyUnicode_GetLength(qualities_obj);
    Py_ssize_t qualities_length = 0;
    const uint8_t *qualities = (const uint8_t *)PyUnicode_AsUTF8AndSize(
        qualities_obj, &qualities_length);
    if (original_qualities_length != qualities_length) {
        PyErr_Format(PyExc_ValueError,
                     "qualities should contain only ASCII characters: %R",
                     sequence_obj);
        return NULL;
    }

    if (sequence_length != qualities_length) {
        PyErr_Format(
            PyExc_ValueError,
            "sequence and qualities have different lengths: %zd and %zd",
            sequence_length, qualities_length);
        return NULL;
    }
    Py_ssize_t tags_length = 0;
    char *tags = NULL;
    if (tags_obj != NULL) {
        tags_length = PyBytes_Size(tags_obj);
        tags = PyBytes_AsString(tags_obj);
    }
    size_t total_length = name_length + sequence_length * 2 + tags_length;
    if (total_length > UINT32_MAX) {
        // lengths are saved as uint32_t types so throw an error;
        PyErr_Format(
            PyExc_OverflowError,
            "Total length of FASTQ record exceeds 4 GiB. Record name: %R",
            name_obj);
        return NULL;
    }

    double accumulated_error_rate = 0.0;
    for (Py_ssize_t i = 0; i < sequence_length; i++) {
        uint8_t q = qualities[i] - 33;
        if (q > PHRED_MAX) {
            PyErr_Format(PyExc_ValueError, "Not a valid phred character: %c",
                         qualities[i]);
            return NULL;
        }
        accumulated_error_rate += SCORE_TO_ERROR_RATE[q];
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

    uint8_t *buffer = (uint8_t *)PyBytes_AsString(bytes_obj);  // No check needed.
    self->meta.record_start = buffer;
    self->meta.name_length = name_length;
    self->meta.sequence_offset = name_length;
    self->meta.sequence_length = sequence_length;
    self->meta.qualities_offset = name_length + sequence_length;
    self->meta.tags_offset = name_length + sequence_length * 2;
    self->meta.tags_length = tags_length;
    self->meta.accumulated_error_rate = accumulated_error_rate;
    self->obj = bytes_obj;

    memcpy(buffer, name, name_length);
    size_t cursor = name_length;
    memcpy(buffer + cursor, sequence, sequence_length);
    cursor += sequence_length;
    memcpy(buffer + cursor, qualities, sequence_length);
    cursor += sequence_length;
    memcpy(buffer + cursor, tags, tags_length);
    return (PyObject *)self;
}

PyDoc_STRVAR(FastqRecordView_name__doc__,
             "name($self)\n"
             "--\n"
             "\n"
             "Returns the FASTQ header.\n");

static PyObject *
FastqRecordView_name(FastqRecordView *self, PyObject *Py_UNUSED(ignore))
{
    return PyUnicode_DecodeASCII((char *)self->meta.name,
                                 self->meta.name_length, NULL);
}

PyDoc_STRVAR(FastqRecordView_sequence__doc__,
             "sequence($self)\n"
             "--\n"
             "\n"
             "Returns the FASTQ nucleotide sequence.\n");

static PyObject *
FastqRecordView_sequence(FastqRecordView *self, PyObject *Py_UNUSED(ignore))
{
    return PyUnicode_DecodeASCII(
        (char *)self->meta.record_start + self->meta.sequence_offset,
        self->meta.sequence_length, NULL);
}

PyDoc_STRVAR(FastqRecordView_qualities__doc__,
             "qualities($self)\n"
             "--\n"
             "\n"
             "Returns the FASTQ phred encoded qualities as a string.\n");

static PyObject *
FastqRecordView_qualities(FastqRecordView *self, PyObject *Py_UNUSED(ignore))
{
    return PyUnicode_DecodeASCII(
        (char *)self->meta.record_start + self->meta.qualities_offset,
        self->meta.sequence_length, NULL);
}

PyDoc_STRVAR(FastqRecordView_tags__doc__,
             "tags($self)\n"
             "--\n"
             "\n"
             "Returns the raw tags as a bytes object.\n");
static PyObject *
FastqRecordView_tags(FastqRecordView *self, PyObject *Py_UNUSED(ignore))
{
    return PyBytes_FromStringAndSize(
        (char *)self->meta.record_start + self->meta.tags_offset,
        self->meta.tags_length);
}

static PyMethodDef FastqRecordView_methods[] = {
    {"name", (PyCFunction)FastqRecordView_name, METH_NOARGS,
     FastqRecordView_name__doc__},
    {"sequence", (PyCFunction)FastqRecordView_sequence, METH_NOARGS,
     FastqRecordView_sequence__doc__},
    {"qualities", (PyCFunction)FastqRecordView_qualities, METH_NOARGS,
     FastqRecordView_qualities__doc__},
    {"tags", (PyCFunction)FastqRecordView_tags, METH_NOARGS,
     FastqRecordView_tags__doc__},
    {NULL}};

static PyMemberDef FastqRecordView_members[] = {
    {"obj", T_OBJECT, offsetof(FastqRecordView, obj), READONLY,
     "The underlying buffer where the fastq record is located"},
    {NULL},
};

static PyType_Slot FastqRecordView_slots[] = {
    {Py_tp_dealloc, (destructor)FastqRecordView_dealloc},
    {Py_tp_new, (newfunc)FastqRecordView__new__},
    {Py_tp_methods, FastqRecordView_methods},
    {Py_tp_members, FastqRecordView_members},
    {0, NULL},
};

static PyType_Spec FastqRecordView_spec = {
    .name = "_qc.FastqRecordView",
    .basicsize = sizeof(FastqRecordView),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT,
    .slots = FastqRecordView_slots,
};

/************************
 * FastqRecordArrayView *
 ************************/

typedef struct _FastqRecordArrayViewStruct {
    PyObject_VAR_HEAD
    PyObject *obj;
    struct FastqMeta records[];
} FastqRecordArrayView;

static void
FastqRecordArrayView_dealloc(FastqRecordArrayView *self)
{
    Py_XDECREF(self->obj);
    PyObject *tp = (PyObject *)Py_TYPE((PyObject *)self);
    PyObject_Free(self);
    Py_DECREF(tp);
}

static PyObject *
FastqRecordArrayView_FromPointerSizeAndObject(struct FastqMeta *records,
                                              size_t number_of_records,
                                              PyObject *obj,
                                              PyTypeObject *FastqRecordArrayView_Type)
{
    size_t size = number_of_records * sizeof(struct FastqMeta);
    FastqRecordArrayView *self = PyObject_NewVar(
        FastqRecordArrayView, FastqRecordArrayView_Type, number_of_records);
    if (self == NULL) {
        return PyErr_NoMemory();
    }
    if (records != NULL) {
        memcpy(self->records, records, size);
    }
    Py_INCREF(obj);
    self->obj = obj;
    return (PyObject *)self;
}

static PyObject *
FastqRecordArrayView__new__(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyObject *view_items_obj = NULL;
    static char *format = "O:FastqRecordArrayView";
    static char *kwargnames[] = {"view_items", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, kwargnames,
                                     &view_items_obj)) {
        return NULL;
    }
    PyObject *view_fastseq =
        PySequence_Fast(view_items_obj, "view_items should be iterable");
    if (view_fastseq == NULL) {
        return NULL;
    }
    Py_ssize_t number_of_items = PySequence_Length(view_fastseq);

    size_t total_memory_size = 0;
    struct QCModuleState *qc_module_state = get_qc_module_state_from_type(type);
    PyTypeObject *FastqRecordView_Type = qc_module_state->FastqRecordView_Type;

    for (Py_ssize_t i = 0; i < number_of_items; i++) {
        PyObject *item = PySequence_GetItem(view_fastseq, i);
        int correct_type =
            PyObject_IsInstance(item, (PyObject *)FastqRecordView_Type);
        if (correct_type == -1) {
            Py_DECREF(item);
            return NULL;
        }
        else if (correct_type == 0) {
            PyErr_Format(
                PyExc_TypeError,
                "Expected an iterable of FastqRecordView objects, but item %z "
                "is of type %R: %R",
                i, (PyObject *)Py_TYPE((PyObject *)item), item);
            Py_DECREF(item);
            return NULL;
        }
        FastqRecordView *record = (FastqRecordView *)item;
        size_t memory_size = record->meta.name_length +
                             record->meta.sequence_length * 2 +
                             record->meta.tags_length;
        total_memory_size += memory_size;
        Py_DECREF(item);
    }
    PyObject *obj = PyBytes_FromStringAndSize(NULL, total_memory_size);
    if (obj == NULL) {
        return PyErr_NoMemory();
    }
    FastqRecordArrayView *record_array =
        (FastqRecordArrayView *)FastqRecordArrayView_FromPointerSizeAndObject(
            NULL, number_of_items, obj,
            qc_module_state->FastqRecordArrayView_Type);
    Py_DECREF(obj);  // Reference count increased by 1 by previous function.
    if (record_array == NULL) {
        Py_DECREF(obj);
        return NULL;
    }
    char *record_ptr = PyBytes_AsString(obj);  // No check needed.
    struct FastqMeta *metas = record_array->records;
    for (Py_ssize_t i = 0; i < number_of_items; i++) {
        FastqRecordView *record =
            (FastqRecordView *)PySequence_GetItem(view_fastseq, i);
        struct FastqMeta meta = record->meta;
        memcpy(record_ptr, meta.record_start, meta.name_length);
        record_ptr += meta.name_length;
        memcpy(record_ptr, meta.record_start + meta.sequence_offset,
               meta.sequence_length);
        record_ptr += meta.sequence_length;
        memcpy(record_ptr, meta.record_start + meta.qualities_offset,
               meta.sequence_length);
        record_ptr += meta.sequence_length;
        memcpy(record_ptr, meta.record_start + meta.tags_offset, meta.tags_length);
        record_ptr += meta.tags_length;
        memcpy(metas + i, &record->meta, sizeof(struct FastqMeta));
        Py_DECREF(record);
    }
    return (PyObject *)record_array;
}

static PyObject *
FastqRecordArrayView__get_item__(FastqRecordArrayView *self, Py_ssize_t i)
{
    Py_ssize_t size = Py_SIZE((PyObject *)self);
    if (i < 0) {
        i = size + i;
    }
    if (i < 0 || i >= size) {
        PyErr_SetString(PyExc_IndexError, "array index out of range");
        return NULL;
    }
    struct QCModuleState *qc_module_state = get_qc_module_state_from_obj(self);
    if (qc_module_state == NULL) {
        return NULL;
    }
    FastqRecordView *view =
        PyObject_New(FastqRecordView, qc_module_state->FastqRecordView_Type);
    if (self == NULL) {
        return PyErr_NoMemory();
    }
    memcpy(&view->meta, self->records + i, sizeof(struct FastqMeta));
    Py_XINCREF(self->obj);
    view->obj = self->obj;
    return (PyObject *)view;
}

static inline Py_ssize_t
FastqRecordArrayView__length__(FastqRecordArrayView *self)
{
    return Py_SIZE((PyObject *)self);
}

static inline bool
is_space(char c)
{
    return (c == ' ' || c == '\t');
}

#define FIND_SPACE_CHUNK_SIZE 8
/**
 * @brief strcspn(str, " \t") replacement with length. Returns the offset of
 * ' ' or '\t' or the length of the string.
 *
 * @param str
 * @param length
 * @return size_t
 */
static inline size_t
find_space(const char *restrict str, size_t length)
{
    const char *restrict cursor = str;
    const char *end = str + length;
    const char *vec_end = end - (FIND_SPACE_CHUNK_SIZE - 1);
    while (cursor < vec_end) {
        /* Fixed size for loop allows compiler to use inline vectors. */
        uint8_t results[FIND_SPACE_CHUNK_SIZE];
        for (size_t i = 0; i < FIND_SPACE_CHUNK_SIZE; i++) {
            /* Set all bits when is_space is true. This is the same result as
               when _mm_cmpeq_epi8 is used. Hence an extra AND instruction is
               prevented. */
            results[i] = is_space(cursor[i]) ? 255 : 0;
        }
        uint64_t *result = (uint64_t *)results;
        if (result[0]) {
            break;
        }
        cursor += 8;
    }
    while (cursor < end) {
        if (is_space(*cursor)) {
            break;
        }
        cursor++;
    }
    return cursor - str;
}

/**
 * @brief Compare two FASTQ record names to see if they are mates.
 *
 * They are mates if their IDs are the same. The ID is the part before the
 * first whitespace. If the last symbol of both IDs is a '1' or '2' it is
 * ignored to allow 'record/1' and 'record/2' to match.
 *
 * @param name1 Pointer to the first name
 * @param name2 Pointer to the second name
 * @param name2_length The length of the second name
 */
static inline bool
fastq_names_are_mates(const char *name1, const char *name2,
                      size_t name1_length, size_t name2_length)
{
    size_t id_length = find_space(name1, name1_length);
    if (name2_length < id_length) {
        return false;
    }
    if (name2_length > id_length) {
        char id2_sep = name2[id_length];
        if (!(id2_sep == ' ' || id2_sep == '\t')) {
            return false;
        }
    }
    /* Make sure /1 and /2 endings are ignored. */
    char id1_last_char = name1[id_length - 1];
    if (id1_last_char == '1' || id1_last_char == '2') {
        char id2_last_char = name2[id_length - 1];
        if (id2_last_char == '1' || id2_last_char == '2') {
            id_length -= 1;
        }
    }
    return memcmp(name1, name2, id_length) == 0;
}

PyDoc_STRVAR(
    FastqRecordArrayView_is_mate__doc__,
    "is_mate($self, other, /)\n"
    "--\n"
    "\n"
    "Check if the record IDs in this array view match with those in other.\n"
    "\n"
    "  other\n"
    "    Another FastqRecordArrayView object.\n");

#define FastqRecordArrayView_is_mate_method METH_O
static PyObject *
FastqRecordArrayView_is_mate(FastqRecordArrayView *self, PyObject *other_obj)
{
    int instance_check =
        PyObject_IsInstance(other_obj, (PyObject *)Py_TYPE((PyObject *)self));
    if (instance_check == 0) {
        PyErr_Format(PyExc_TypeError,
                     "other must be of type FastqRecordArrayView, got %R",
                     (PyObject *)Py_TYPE((PyObject *)other_obj));
        return NULL;
    }
    else if (instance_check == -1) {
        return NULL;
    }
    FastqRecordArrayView *other = (FastqRecordArrayView *)other_obj;
    Py_ssize_t length = Py_SIZE((PyObject *)self);
    if (length != Py_SIZE((PyObject *)other)) {
        PyErr_Format(PyExc_ValueError,
                     "other is not the same length as this record array view. "
                     "This length: %zd, other length: %zd",
                     length, Py_SIZE((PyObject *)other));
        return NULL;
    }
    struct FastqMeta *self_records = self->records;
    struct FastqMeta *other_records = other->records;
    for (Py_ssize_t i = 0; i < length; i++) {
        struct FastqMeta *record1 = self_records + i;
        struct FastqMeta *record2 = other_records + i;
        char *name1 = (char *)record1->name;
        char *name2 = (char *)record2->name;
        size_t name1_length = record1->name_length;
        size_t name2_length = record2->name_length;
        if (!fastq_names_are_mates(name1, name2, name1_length, name2_length)) {
            Py_RETURN_FALSE;
        }
    }
    Py_RETURN_TRUE;
}

static PyMethodDef FastqRecordArrayView_methods[] = {
    {"is_mate", (PyCFunction)FastqRecordArrayView_is_mate,
     FastqRecordArrayView_is_mate_method, FastqRecordArrayView_is_mate__doc__},
    {NULL},
};

static PyMemberDef FastqRecordArrayView_members[] = {
    {"obj", T_OBJECT, offsetof(FastqRecordArrayView, obj), READONLY,
     "The underlying buffer where the fastq records are located"},
    {NULL},
};

static PyType_Slot FastqRecordArrayView_slots[] = {
    {Py_sq_item, (ssizeargfunc)FastqRecordArrayView__get_item__},
    {Py_sq_length, (lenfunc)FastqRecordArrayView__length__},
    {Py_tp_members, FastqRecordArrayView_members},
    {Py_tp_methods, FastqRecordArrayView_methods},
    {Py_tp_new, FastqRecordArrayView__new__},
    {
        Py_tp_dealloc,
        (destructor)FastqRecordArrayView_dealloc,
    },
    {0, NULL},
};

static PyType_Spec FastqRecordArrayView_spec = {
    .name = "_qc.FastqRecordArrayView",
    .basicsize = sizeof(FastqRecordArrayView),
    .itemsize = sizeof(struct FastqMeta),
    .flags = Py_TPFLAGS_DEFAULT,
    .slots = FastqRecordArrayView_slots,
};

/****************
 * FASTQ PARSER *
 ****************/

typedef struct _FastqParserStruct {
    PyObject_HEAD
    uint8_t *record_start;
    uint8_t *buffer_end;
    size_t read_in_size;
    PyObject *buffer_obj;
    struct FastqMeta *meta_buffer;
    size_t meta_buffer_size;
    PyObject *file_obj;
} FastqParser;

static void
FastqParser_dealloc(FastqParser *self)
{
    Py_XDECREF(self->buffer_obj);
    Py_XDECREF(self->file_obj);
    PyMem_Free(self->meta_buffer);
    PyTypeObject *tp = Py_TYPE((PyObject *)self);
    PyObject_Free(self);
    Py_XDECREF((PyObject *)tp);
}

static PyObject *
FastqParser__new__(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyObject *file_obj = NULL;
    size_t read_in_size = 128 * 1024;
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
    self->record_start = (uint8_t *)PyBytes_AsString(buffer_obj);
    self->buffer_end = self->record_start;
    self->buffer_obj = buffer_obj;
    self->read_in_size = read_in_size;
    self->meta_buffer = NULL;
    self->meta_buffer_size = 0;
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

static inline bool
buffer_contains_fastq(const uint8_t *buffer, size_t buffer_size)
{
    const uint8_t *buffer_end = buffer + buffer_size;
    const uint8_t *buffer_pos = buffer;
    /* Four newlines should be present in a FASTQ record. */
    for (size_t i = 0; i < 4; i++) {
        buffer_pos = memchr(buffer_pos, '\n', buffer_end - buffer_pos);
        if (buffer_pos == NULL) {
            return false;
        }
        /* Skip the found newline */
        buffer_pos += 1;
    }
    return true;
}

static PyObject *
FastqParser_create_record_array(FastqParser *self, size_t min_records,
                                size_t max_records)
{
    struct QCModuleState *state = get_qc_module_state_from_obj(self);
    PyTypeObject *FastqRecordArrayView_Type = state->FastqRecordArrayView_Type;

    uint8_t *record_start = self->record_start;
    uint8_t *buffer_end = self->buffer_end;
    size_t parsed_records = 0;
    PyObject *new_buffer_obj = NULL;

    while (parsed_records < min_records) {
        size_t leftover_size = buffer_end - record_start;
        size_t read_in_size;
        size_t read_in_offset;
        Py_ssize_t new_buffer_size;
        size_t record_start_offset;
        if (new_buffer_obj == NULL) {
            /* On the first loop create a new buffer and initialize it with
               the leftover from the last run of the function. */
            new_buffer_size = self->read_in_size;
            new_buffer_obj = PyBytes_FromStringAndSize(NULL, new_buffer_size);
            if (new_buffer_obj == NULL) {
                return NULL;
            }
            memcpy(PyBytes_AsString(new_buffer_obj), record_start, leftover_size);
            read_in_size = new_buffer_size - leftover_size;
            read_in_offset = leftover_size;
            record_start_offset = 0;
        }
        else {
            /* On subsequent loops, enlarge the buffer until the minimum
               amount of records fits. */
            PyObject *older_buffer_obj = new_buffer_obj;
            uint8_t *old_start = (uint8_t *)PyBytes_AsString(older_buffer_obj);
            record_start_offset = record_start - old_start;
            size_t old_size = buffer_end - old_start;
            new_buffer_size = old_size + self->read_in_size;
            new_buffer_obj = PyBytes_FromStringAndSize(NULL, new_buffer_size);
            if (new_buffer_obj == NULL) {
                return NULL;
            }
            uint8_t *new_start = (uint8_t *)PyBytes_AsString(new_buffer_obj);
            memcpy(new_start, old_start, old_size);
            Py_DECREF(older_buffer_obj);

            /* Change the already parsed records to point to the new buffer. */
            struct FastqMeta *meta_buffer = self->meta_buffer;
            for (size_t i = 0; i < parsed_records; i++) {
                struct FastqMeta *record = meta_buffer + i;
                intptr_t record_offset = record->record_start - old_start;
                record->record_start = new_start + record_offset;
            }
            read_in_size = self->read_in_size;
            read_in_offset = old_size;
        }
        uint8_t *new_buffer = (uint8_t *)PyBytes_AsString(new_buffer_obj);

        PyObject *remaining_space_view = PyMemoryView_FromMemory(
            (char *)new_buffer + read_in_offset, read_in_size, PyBUF_WRITE);
        if (remaining_space_view == NULL) {
            return NULL;
        }
        PyObject *read_bytes_obj = PyObject_CallMethod(
            self->file_obj, "readinto", "O", remaining_space_view);
        Py_DECREF(remaining_space_view);
        if (read_bytes_obj == NULL) {
            Py_DECREF(new_buffer_obj);
            return NULL;
        }
        Py_ssize_t read_bytes = PyLong_AsSsize_t(read_bytes_obj);
        if (read_bytes == -1) {
            Py_DECREF(new_buffer_obj);
            return NULL;
        }
        Py_DECREF(read_bytes_obj);
        Py_ssize_t actual_buffer_size = read_in_offset + read_bytes;
        if (actual_buffer_size < new_buffer_size) {
            PyObject *old_buffer_obj = new_buffer_obj;
            new_buffer_obj = PyBytes_FromStringAndSize(NULL, actual_buffer_size);
            if (new_buffer_obj == NULL) {
                Py_DECREF(old_buffer_obj);
                return NULL;
            }
            memcpy(PyBytes_AsString(new_buffer_obj),
                   PyBytes_AsString(old_buffer_obj), actual_buffer_size);
            Py_DECREF(old_buffer_obj);
        }
        new_buffer = (uint8_t *)PyBytes_AsString(new_buffer_obj);
        new_buffer_size = actual_buffer_size;
        if (!string_is_ascii((char *)new_buffer + read_in_offset, read_bytes)) {
            Py_ssize_t pos;
            for (pos = read_in_offset; pos < new_buffer_size; pos += 1) {
                if (new_buffer[pos] & ASCII_MASK_1BYTE) {
                    break;
                }
            }
            PyErr_Format(PyExc_ValueError,
                         "Found non-ASCII character in file: %c",
                         new_buffer[pos]);
            Py_DECREF(new_buffer_obj);
            return NULL;
        }

        if (new_buffer_size == 0) {
            // Entire file is read.
            break;
        }
        else if (read_bytes == 0) {
            if (!buffer_contains_fastq(new_buffer, new_buffer_size)) {
                // Incomplete record at the end of file;
                PyErr_Format(PyExc_EOFError,
                             "Incomplete record at the end of file %s",
                             new_buffer);
                Py_DECREF(new_buffer_obj);
                return NULL;
            }
            if (parsed_records) {
                /* min_records not yet reached, but file contains no more
                   records */
                break;
            }
            /* At this point, there are still valid FASTQ records in the buffer
               but these have not been parsed yet.*/
        }
        record_start = new_buffer + record_start_offset;
        buffer_end = new_buffer + new_buffer_size;

        while (parsed_records < max_records) {
            if (record_start + 2 >= buffer_end) {
                break;
            }
            if (record_start[0] != '@') {
                PyErr_Format(PyExc_ValueError,
                             "Record does not start with @ but with %c",
                             record_start[0]);
                Py_DECREF(new_buffer_obj);
                return NULL;
            }
            uint8_t *name_start = record_start + 1;
            uint8_t *name_end =
                memchr(name_start, '\n', buffer_end - record_start);
            if (name_end == NULL) {
                break;
            }
            size_t name_length = name_end - name_start;
            uint8_t *sequence_start = name_end + 1;
            uint8_t *sequence_end =
                memchr(sequence_start, '\n', buffer_end - sequence_start);
            if (sequence_end == NULL) {
                break;
            }
            size_t sequence_length = sequence_end - sequence_start;
            uint8_t *second_header_start = sequence_end + 1;
            if ((second_header_start < buffer_end) &&
                second_header_start[0] != '+') {
                PyErr_Format(
                    PyExc_ValueError,
                    "Record second header does not start with + but with %c",
                    second_header_start[0]);
                Py_DECREF(new_buffer_obj);
                return NULL;
            }
            uint8_t *second_header_end = memchr(
                second_header_start, '\n', buffer_end - second_header_start);
            if (second_header_end == NULL) {
                break;
            }
            uint8_t *qualities_start = second_header_end + 1;
            uint8_t *qualities_end =
                memchr(qualities_start, '\n', buffer_end - qualities_start);
            if (qualities_end == NULL) {
                break;
            }
            size_t qualities_length = qualities_end - qualities_start;
            if (sequence_length != qualities_length) {
                PyObject *record_name_obj =
                    PyUnicode_DecodeASCII((char *)name_start, name_length, NULL);
                PyErr_Format(PyExc_ValueError,
                             "Record sequence and qualities do not have equal "
                             "length, %R",
                             record_name_obj);
                Py_DECREF(new_buffer_obj);
                Py_DECREF(record_name_obj);
                return NULL;
            }
            parsed_records += 1;
            if (parsed_records > self->meta_buffer_size) {
                struct FastqMeta *tmp = PyMem_Realloc(
                    self->meta_buffer, sizeof(struct FastqMeta) * parsed_records);
                if (tmp == NULL) {
                    return PyErr_NoMemory();
                }
                self->meta_buffer = tmp;
                self->meta_buffer_size = parsed_records;
            }
            struct FastqMeta *meta = self->meta_buffer + (parsed_records - 1);
            meta->record_start = name_start;
            meta->name_length = name_length;
            meta->sequence_offset = sequence_start - name_start;
            meta->sequence_length = sequence_length;
            meta->qualities_offset = qualities_start - name_start;
            meta->tags_offset = qualities_end - name_start;
            meta->tags_length = 0;
            meta->accumulated_error_rate = 0.0;
            record_start = qualities_end + 1;
        }
    }
    /* Save the current buffer object so any leftovers can be reused at the
       next invocation. */
    PyObject *tmp = self->buffer_obj;
    self->buffer_obj = new_buffer_obj;
    Py_DECREF(tmp);
    /* Save record start and buffer end for next invocation. */
    self->record_start = record_start;
    self->buffer_end = buffer_end;
    return FastqRecordArrayView_FromPointerSizeAndObject(
        self->meta_buffer, parsed_records, new_buffer_obj,
        FastqRecordArrayView_Type);
}

static PyObject *
FastqParser__next__(FastqParser *self)
{
    PyObject *ret = FastqParser_create_record_array(self, 1, SIZE_MAX);
    if ((ret != NULL) && (Py_SIZE((PyObject *)ret) == 0)) {
        PyErr_SetNone(PyExc_StopIteration);
        Py_DECREF(ret);
        return NULL;
    }
    return ret;
}

PyDoc_STRVAR(FastqParser_read__doc__,
             "read($self, number_of_records, /)\n"
             "--\n"
             "\n"
             "Read a number of records into a count array.\n"
             "\n"
             "  Number_of_records\n"
             "    Number_of_records that should be attempted to read.\n");

#define FastqParser_read_method METH_O

static PyObject *
FastqParser_read(FastqParser *self, PyObject *number_of_records_obj)
{
    Py_ssize_t number_of_records = PyLong_AsSsize_t(number_of_records_obj);
    if (number_of_records < 1) {
        PyErr_Format(PyExc_ValueError,
                     "number_of_records should be greater than 1, got %zd",
                     number_of_records);
        return NULL;
    }
    return FastqParser_create_record_array(self, number_of_records,
                                           number_of_records);
}

static PyMethodDef FastqParser_methods[] = {
    {"read", (PyCFunction)FastqParser_read, FastqParser_read_method,
     FastqParser_read__doc__},
    {NULL},
};

static PyType_Slot FastqParser_slots[] = {
    {Py_tp_dealloc, (destructor)FastqParser_dealloc},
    {Py_tp_new, FastqParser__new__},
    {Py_tp_iter, FastqParser__iter__},
    {Py_tp_iternext, FastqParser__next__},
    {Py_tp_methods, FastqParser_methods},
    {0, NULL},
};

static PyType_Spec FastqParser_spec = {
    .name = "_qc.FastqParser",
    .basicsize = sizeof(FastqParser),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT,
    .slots = FastqParser_slots,
};

/**************
 * BAM PARSER *
 * ************/

#define BAM_FPAIRED 1
#define BAM_FPROPER_PAIR 2
#define BAM_FUNMAP 4
#define BAM_FMUNMAP 8
#define BAM_FREVERSE 16
#define BAM_FMREVERSE 32
#define BAM_FREAD1 64
#define BAM_FREAD2 128
#define BAM_FSECONDARY 256
#define BAM_FQCFAIL 512
#define BAM_FDUP 1024
#define BAM_FSUPPLEMENTARY 2048

#define BAM_EXCLUDE_FLAGS (BAM_FSECONDARY | BAM_FSUPPLEMENTARY)

static void
decode_bam_sequence_default(uint8_t *dest, const uint8_t *encoded_sequence,
                            size_t length)
{
    static const char code2base[512] =
        "===A=C=M=G=R=S=V=T=W=Y=H=K=D=B=N"
        "A=AAACAMAGARASAVATAWAYAHAKADABAN"
        "C=CACCCMCGCRCSCVCTCWCYCHCKCDCBCN"
        "M=MAMCMMMGMRMSMVMTMWMYMHMKMDMBMN"
        "G=GAGCGMGGGRGSGVGTGWGYGHGKGDGBGN"
        "R=RARCRMRGRRRSRVRTRWRYRHRKRDRBRN"
        "S=SASCSMSGSRSSSVSTSWSYSHSKSDSBSN"
        "V=VAVCVMVGVRVSVVVTVWVYVHVKVDVBVN"
        "T=TATCTMTGTRTSTVTTTWTYTHTKTDTBTN"
        "W=WAWCWMWGWRWSWVWTWWWYWHWKWDWBWN"
        "Y=YAYCYMYGYRYSYVYTYWYYYHYKYDYBYN"
        "H=HAHCHMHGHRHSHVHTHWHYHHHKHDHBHN"
        "K=KAKCKMKGKRKSKVKTKWKYKHKKKDKBKN"
        "D=DADCDMDGDRDSDVDTDWDYDHDKDDDBDN"
        "B=BABCBMBGBRBSBVBTBWBYBHBKBDBBBN"
        "N=NANCNMNGNRNSNVNTNWNYNHNKNDNBNN";
    static const uint8_t *nuc_lookup = (uint8_t *)"=ACMGRSVTWYHKDBN";
    size_t length_2 = length / 2;
    for (size_t i = 0; i < length_2; i++) {
        memcpy(dest + i * 2, code2base + ((size_t)encoded_sequence[i] * 2), 2);
    }
    if (length & 1) {
        uint8_t encoded = encoded_sequence[length_2] >> 4;
        dest[(length - 1)] = nuc_lookup[encoded];
    }
}

static void (*decode_bam_sequence)(uint8_t *dest, const uint8_t *encoded_sequence,
                                   size_t length) = decode_bam_sequence_default;

#if COMPILER_HAS_TARGETED_DISPATCH && BUILD_IS_X86_64
__attribute__((__target__("ssse3"))) static void
decode_bam_sequence_ssse3(uint8_t *dest, const uint8_t *encoded_sequence,
                          size_t length)
{
    static const uint8_t *nuc_lookup = (uint8_t *)"=ACMGRSVTWYHKDBN";
    const uint8_t *dest_end_ptr = dest + length;
    uint8_t *dest_cursor = dest;
    const uint8_t *encoded_cursor = encoded_sequence;
    const uint8_t *dest_vec_end_ptr = dest_end_ptr - (2 * sizeof(__m128i) - 1);
    __m128i nuc_lookup_vec = _mm_lddqu_si128((__m128i *)nuc_lookup);
    /* Nucleotides are encoded 4-bits per nucleotide and stored in 8-bit bytes
       as follows: |AB|CD|EF|GH|. The 4-bit codes (going from 0-15) can be used
       together with the pshufb instruction as a lookup table. The most
       efficient way is to use bitwise AND and shift to create two vectors. One
       with all the upper codes (|A|C|E|G|) and one with the lower codes
       (|B|D|F|H|). The lookup can then be performed and the resulting vectors
       can be interleaved again using the unpack instructions. */
    while (dest_cursor < dest_vec_end_ptr) {
        __m128i encoded = _mm_lddqu_si128((__m128i *)encoded_cursor);
        __m128i encoded_upper = _mm_srli_epi64(encoded, 4);
        encoded_upper = _mm_and_si128(encoded_upper, _mm_set1_epi8(15));
        __m128i encoded_lower = _mm_and_si128(encoded, _mm_set1_epi8(15));
        __m128i nucs_upper = _mm_shuffle_epi8(nuc_lookup_vec, encoded_upper);
        __m128i nucs_lower = _mm_shuffle_epi8(nuc_lookup_vec, encoded_lower);
        __m128i first_nucleotides = _mm_unpacklo_epi8(nucs_upper, nucs_lower);
        __m128i second_nucleotides = _mm_unpackhi_epi8(nucs_upper, nucs_lower);
        _mm_storeu_si128((__m128i *)dest_cursor, first_nucleotides);
        _mm_storeu_si128((__m128i *)(dest_cursor + 16), second_nucleotides);
        encoded_cursor += sizeof(__m128i);
        dest_cursor += 2 * sizeof(__m128i);
    }
    decode_bam_sequence_default(dest_cursor, encoded_cursor,
                                dest_end_ptr - dest_cursor);
}

/* Constructor runs at dynamic load time */
__attribute__((constructor)) static void
decode_bam_sequence_init_func_ptr(void)
{
    if (__builtin_cpu_supports("ssse3")) {
        decode_bam_sequence = decode_bam_sequence_ssse3;
    }
    else {
        decode_bam_sequence = decode_bam_sequence_default;
    }
}
#endif

// Code is simple enough to be auto vectorized.
#if COMPILER_HAS_OPTIMIZE
__attribute__((optimize("O3")))
#endif
static void
decode_bam_qualities(uint8_t *restrict dest,
                     const uint8_t *restrict encoded_qualities, size_t length)
{
    for (size_t i = 0; i < length; i++) {
        dest[i] = encoded_qualities[i] + 33;
    }
}

typedef struct _BamParserStruct {
    PyObject_HEAD
    uint8_t *record_start;
    uint8_t *buffer_end;
    size_t read_in_size;
    uint8_t *read_in_buffer;
    size_t read_in_buffer_size;
    struct FastqMeta *meta_buffer;
    size_t meta_buffer_size;
    PyObject *file_obj;
    PyObject *header;  // The BAM header
} BamParser;

static void
BamParser_dealloc(BamParser *self)
{
    PyMem_Free(self->read_in_buffer);
    PyMem_Free(self->meta_buffer);
    Py_XDECREF(self->file_obj);
    Py_XDECREF(self->header);
    PyTypeObject *tp = Py_TYPE((PyObject *)self);
    PyObject_Free(self);
    Py_XDECREF((PyObject *)tp);
}

static PyObject *
BamParser__new__(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyObject *file_obj = NULL;
    size_t read_in_size = 48 * 1024;  // Slightly smaller than BGZF block size
    static char *kwargnames[] = {"fileobj", "initial_buffersize", NULL};
    static char *format = "O|n:BamParser";
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, kwargnames,
                                     &file_obj, &read_in_size)) {
        return NULL;
    }
    if (read_in_size < 4) {
        // At least 4 so the block_size of a new record can be immediately read.
        PyErr_Format(PyExc_ValueError,
                     "initial_buffersize must be at least 4, got %zd",
                     read_in_size);
        return NULL;
    }

    /*Read BAM file header and skip ahead to the records */
    PyObject *magic_and_header_size =
        PyObject_CallMethod(file_obj, "read", "n", 8);
    if (magic_and_header_size == NULL) {
        return NULL;
    }
    if (!PyBytes_CheckExact(magic_and_header_size)) {
        PyErr_Format(PyExc_TypeError,
                     "file_obj %R is not a binary IO type, got %R", file_obj,
                     Py_TYPE((PyObject *)file_obj));
        Py_DECREF(magic_and_header_size);
        return NULL;
    }
    if (PyBytes_Size(magic_and_header_size) < 8) {
        PyErr_SetString(PyExc_EOFError, "Truncated BAM file");
        Py_DECREF(magic_and_header_size);
        return NULL;
    }
    uint8_t *file_start = (uint8_t *)PyBytes_AsString(magic_and_header_size);
    if (memcmp(file_start, "BAM\1", 4) != 0) {
        PyErr_Format(
            PyExc_ValueError,
            "fileobj: %R, is not a BAM file. No BAM magic, instead found: %R",
            file_obj, magic_and_header_size);
        Py_DECREF(magic_and_header_size);
        return NULL;
    }
    uint32_t l_text = *(uint32_t *)(file_start + 4);
    Py_DECREF(magic_and_header_size);
    PyObject *header = PyObject_CallMethod(file_obj, "read", "n", l_text);
    if (PyBytes_Size(header) != l_text) {
        PyErr_SetString(PyExc_EOFError, "Truncated BAM file");
        Py_DECREF(header);
        return NULL;
    }
    PyObject *n_ref_obj = PyObject_CallMethod(file_obj, "read", "n", 4);
    if (PyBytes_Size(n_ref_obj) != 4) {
        PyErr_SetString(PyExc_EOFError, "Truncated BAM file");
        Py_DECREF(n_ref_obj);
        Py_DECREF(header);
        return NULL;
    }
    uint32_t n_ref = *(uint32_t *)PyBytes_AsString(n_ref_obj);
    Py_DECREF(n_ref_obj);

    for (size_t i = 0; i < n_ref; i++) {
        PyObject *l_name_obj = PyObject_CallMethod(file_obj, "read", "n", 4);
        if (PyBytes_Size(l_name_obj) != 4) {
            PyErr_SetString(PyExc_EOFError, "Truncated BAM file");
            Py_DECREF(header);
            return NULL;
        }
        size_t l_name = *(uint32_t *)PyBytes_AsString(l_name_obj);
        Py_DECREF(l_name_obj);
        Py_ssize_t reference_chunk_size =
            l_name + 4;  // Includes name and uint32_t for size.
        PyObject *reference_chunk =
            PyObject_CallMethod(file_obj, "read", "n", reference_chunk_size);
        Py_ssize_t actual_reference_chunk_size = PyBytes_Size(reference_chunk);
        Py_DECREF(reference_chunk);
        if (actual_reference_chunk_size != reference_chunk_size) {
            PyErr_SetString(PyExc_EOFError, "Truncated BAM file");
            Py_DECREF(header);
            return NULL;
        }
    }
    /* The reader is now skipped ahead to the BAM Records */

    BamParser *self = PyObject_New(BamParser, type);
    if (self == NULL) {
        return PyErr_NoMemory();
    }
    self->read_in_buffer = NULL;
    self->read_in_buffer_size = 0;
    self->buffer_end = self->read_in_buffer;
    self->record_start = self->read_in_buffer;
    self->read_in_size = read_in_size;
    self->meta_buffer = NULL;
    self->meta_buffer_size = 0;
    Py_INCREF(file_obj);
    self->file_obj = file_obj;
    self->header = header;
    return (PyObject *)self;
}

static PyObject *
BamParser__iter__(BamParser *self)
{
    Py_INCREF((PyObject *)self);
    return (PyObject *)self;
}

struct BamRecordHeader {
    uint32_t block_size;
    int32_t reference_id;
    int32_t pos;
    uint8_t l_read_name;
    uint8_t mapq;
    uint16_t bin;
    uint16_t n_cigar_op;
    uint16_t flag;
    uint32_t l_seq;
    int32_t next_ref_id;
    int32_t next_pos;
    int32_t tlen;
};

static PyObject *
BamParser__next__(BamParser *self)
{
    uint8_t *record_start = self->record_start;
    uint8_t *buffer_end = self->buffer_end;
    size_t leftover_size = buffer_end - record_start;
    memmove(self->read_in_buffer, record_start, leftover_size);
    record_start = self->read_in_buffer;
    buffer_end = record_start + leftover_size;
    size_t parsed_records = 0;
    size_t skipped_records = 0;
    PyObject *read_data_obj = NULL;
    struct QCModuleState *state = get_qc_module_state_from_obj(self);
    if (state == NULL) {
        return NULL;
    }
    PyTypeObject *FastqRecordArrayView_Type = state->FastqRecordArrayView_Type;

    while (parsed_records + skipped_records == 0) {
        /* Keep expanding input buffer until at least one record is parsed */
        size_t read_in_size;
        leftover_size = buffer_end - record_start;
        if (leftover_size >= 4) {
            // Immediately check how much the block is to load enough data;
            uint32_t block_size = *(uint32_t *)record_start;
            read_in_size = Py_MAX(block_size, self->read_in_size);
        }
        else {
            // Fill up the buffer up to read_in_size
            read_in_size = self->read_in_size - leftover_size;
        }
        size_t minimum_space_required = leftover_size + read_in_size;
        if (minimum_space_required > self->read_in_buffer_size) {
            uint8_t *tmp_read_in_buffer =
                PyMem_Realloc(self->read_in_buffer, minimum_space_required);
            if (tmp_read_in_buffer == NULL) {
                Py_XDECREF(read_data_obj);
                return PyErr_NoMemory();
            }
            self->read_in_buffer = tmp_read_in_buffer;
            self->read_in_buffer_size = minimum_space_required;
        }
        PyObject *buffer_view =
            PyMemoryView_FromMemory((char *)self->read_in_buffer + leftover_size,
                                    read_in_size, PyBUF_WRITE);
        if (buffer_view == NULL) {
            return NULL;
        }
        PyObject *read_bytes_obj =
            PyObject_CallMethod(self->file_obj, "readinto", "O", buffer_view);
        Py_DECREF(buffer_view);
        if (read_bytes_obj == NULL) {
            Py_XDECREF(read_data_obj);
            return NULL;
        }
        Py_ssize_t read_bytes = PyLong_AsSsize_t(read_bytes_obj);
        Py_DECREF(read_bytes_obj);
        size_t new_buffer_size = leftover_size + read_bytes;
        if (new_buffer_size == 0) {
            // Entire file is read
            PyErr_SetNone(PyExc_StopIteration);
            Py_XDECREF(read_data_obj);
            return NULL;
        }
        else if (read_bytes == 0) {
            PyObject *remaining_obj = PyBytes_FromStringAndSize(
                (char *)self->read_in_buffer, leftover_size);
            PyErr_Format(PyExc_EOFError,
                         "Incomplete record at the end of file %R",
                         remaining_obj);
            Py_DECREF(remaining_obj);
            Py_XDECREF(read_data_obj);
            return NULL;
        }

        record_start = self->read_in_buffer;
        self->record_start = record_start;
        buffer_end = record_start + new_buffer_size;

        /* Bam record consists of name, cigar, sequence, qualities and tags.
           Only space for name, sequence and qualities is needed. Worst case
           scenario space wise is that sequence and qualities are close to 100%
           of the bam record. Quality always maps one to one, but sequence is
           compressed and maps one to two. So that is a 3:4 ratio for BAM:FASTQ.
        */
        Py_ssize_t read_data_size = (new_buffer_size * 4 + 2) / 3;
        if (read_data_obj == NULL) {
            read_data_obj = PyBytes_FromStringAndSize(NULL, read_data_size);
            if (read_data_obj == NULL) {
                Py_XDECREF(read_data_obj);
                return PyErr_NoMemory();
            }
        }
        else {
            PyObject *old_read_data_obj = read_data_obj;
            read_data_obj = PyBytes_FromStringAndSize(NULL, read_data_size);
            if (read_data_obj == NULL) {
                Py_DECREF(old_read_data_obj);
                return NULL;
            }
            uint8_t *read_data_obj_ptr =
                (uint8_t *)PyBytes_AsString(read_data_obj);
            uint8_t *old_read_data_obj_ptr =
                (uint8_t *)PyBytes_AsString(old_read_data_obj);
            Py_ssize_t old_read_data_size = PyBytes_Size(old_read_data_obj);
            memcpy(read_data_obj_ptr, old_read_data_obj_ptr, old_read_data_size);

            /* Adjust FastqMeta relative to the object pointer. */
            for (size_t i = 0; i < parsed_records; i++) {
                struct FastqMeta *meta = self->meta_buffer + i;
                intptr_t offset = meta->record_start - old_read_data_obj_ptr;
                meta->record_start = read_data_obj_ptr + offset;
            }
            Py_DECREF(old_read_data_obj);
        }
        uint8_t *read_data_record_start =
            (uint8_t *)PyBytes_AsString(read_data_obj);

        while (1) {
            if (record_start + 4 >= buffer_end) {
                break;  // Not enough bytes to read block_size
            }
            struct BamRecordHeader *header =
                (struct BamRecordHeader *)record_start;
            uint8_t *record_end = record_start + 4 + header->block_size;
            if (record_end > buffer_end) {
                break;
            }
            if (header->flag & BAM_EXCLUDE_FLAGS) {
                // Skip excluded records such as secondary and supplementary alignments.
                record_start = record_end;
                skipped_records += 1;
                continue;
            }
            uint8_t *bam_name_start =
                record_start + sizeof(struct BamRecordHeader);
            uint32_t name_length = header->l_read_name;
            uint8_t *bam_seq_start = bam_name_start + name_length +
                                     header->n_cigar_op * sizeof(uint32_t);
            uint32_t seq_length = header->l_seq;
            uint32_t encoded_seq_length = (seq_length + 1) / 2;
            uint8_t *bam_qual_start = bam_seq_start + encoded_seq_length;
            uint8_t *tag_start = bam_qual_start + seq_length;
            size_t tags_length = record_end - tag_start;

            uint8_t *read_data_cursor = read_data_record_start;
            if (name_length > 0) {
                name_length -= 1; /* Includes terminating NULL byte */
                memcpy(read_data_cursor, bam_name_start, name_length);
            }
            read_data_cursor += name_length;
            decode_bam_sequence(read_data_cursor, bam_seq_start, seq_length);
            read_data_cursor += seq_length;
            if (seq_length && bam_qual_start[0] == 0xff) {
                /* If qualities are missing, all bases are set to 0xff, which
                   is an invalid phred value. Create a quality string with only
                   zero Phreds for a valid FASTQ representation */
                memset(read_data_cursor, 33, seq_length);
            }
            else {
                decode_bam_qualities(read_data_cursor, bam_qual_start, seq_length);
            }
            read_data_cursor += seq_length;
            memcpy(read_data_cursor, tag_start, tags_length);
            read_data_cursor += tags_length;
            parsed_records += 1;
            if (parsed_records > self->meta_buffer_size) {
                struct FastqMeta *tmp = PyMem_Realloc(
                    self->meta_buffer, sizeof(struct FastqMeta) * parsed_records);
                if (tmp == NULL) {
                    return PyErr_NoMemory();
                }
                self->meta_buffer = tmp;
                self->meta_buffer_size = parsed_records;
            }
            struct FastqMeta *meta = self->meta_buffer + (parsed_records - 1);
            uint32_t sequence_offset = name_length;
            uint32_t qualities_offset = sequence_offset + seq_length;
            uint32_t tags_offset = qualities_offset + seq_length;
            meta->record_start = read_data_record_start;
            meta->name_length = name_length;
            meta->sequence_offset = sequence_offset;
            meta->sequence_length = seq_length;
            meta->qualities_offset = qualities_offset;
            meta->tags_offset = tags_offset;
            meta->tags_length = tags_length;
            meta->accumulated_error_rate = 0.0;
            record_start = record_end;
            read_data_record_start = read_data_cursor;
        }
    }
    self->record_start = record_start;
    self->buffer_end = buffer_end;
    PyObject *record_array = FastqRecordArrayView_FromPointerSizeAndObject(
        self->meta_buffer, parsed_records, read_data_obj,
        FastqRecordArrayView_Type);
    Py_DECREF(read_data_obj);
    return record_array;
}

static PyMemberDef BamParser_members[] = {
    {"header", T_OBJECT_EX, offsetof(BamParser, header), READONLY,
     "The BAM header"},
    {NULL}};

static PyType_Slot BamParser_slots[] = {
    {Py_tp_dealloc, (destructor)BamParser_dealloc},
    {Py_tp_new, BamParser__new__},
    {Py_tp_iter, BamParser__iter__},
    {Py_tp_iternext, BamParser__next__},
    {Py_tp_members, BamParser_members},
    {0, NULL},
};

static PyType_Spec BamParser_spec = {
    .name = "_qc.BamParser",
    .basicsize = sizeof(BamParser),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT,
    .slots = BamParser_slots,
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

#define A 0
#define C 1
#define G 2
#define T 3
#define N 4

// clang-format off
static const uint8_t NUCLEOTIDE_TO_INDEX[128] = {
// Control characters
    N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N,
    N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N,
// Interpunction numbers etc
    N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N,
    N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N,
//     A, B, C, D, E, F, G, H, I, J, K, L, M, N, O,
    N, A, N, C, N, N, N, G, N, N, N, N, N, N, N, N,
//  P, Q, R, S, T, U, V, W, X, Y, Z,  
    N, N, N, N, T, N, N, N, N, N, N, N, N, N, N, N,
//     a, b, c, d, e, f, g, h, i, j, k, l, m, n, o,
    N, A, N, C, N, N, N, G, N, N, N, N, N, N, N, N,
//  p, q, r, s, t, u, v, w, x, y, z, 
    N, N, N, N, T, N, N, N, N, N, N, N, N, N, N, N, 
};
// clang-format on

#define NUC_TABLE_SIZE 5
#define PHRED_LIMIT 47
#define PHRED_TABLE_SIZE ((PHRED_LIMIT / 4) + 1)

#define DEFAULT_END_ANCHOR_LENGTH 100

typedef uint16_t staging_base_table[NUC_TABLE_SIZE];
typedef uint16_t staging_phred_table[PHRED_TABLE_SIZE];
typedef uint64_t base_table[NUC_TABLE_SIZE];
typedef uint64_t phred_table[PHRED_TABLE_SIZE];

static inline uint8_t
phred_to_index(uint8_t phred)
{
    if (phred > PHRED_LIMIT) {
        phred = PHRED_LIMIT;
    }
    return phred >> 2;
}

typedef struct _QCMetricsStruct {
    PyObject_HEAD
    uint8_t phred_offset;
    uint16_t staging_count;
    size_t end_anchor_length;
    size_t max_length;
    staging_base_table *staging_base_counts;
    staging_phred_table *staging_phred_counts;
    staging_base_table *staging_end_anchored_base_counts;
    staging_phred_table *staging_end_anchored_phred_counts;
    base_table *base_counts;
    phred_table *phred_counts;
    base_table *end_anchored_base_counts;
    phred_table *end_anchored_phred_counts;
    size_t number_of_reads;
    uint64_t gc_content[101];
    uint64_t phred_scores[PHRED_MAX + 1];
} QCMetrics;

static void
QCMetrics_dealloc(QCMetrics *self)
{
    PyMem_Free(self->staging_base_counts);
    PyMem_Free(self->staging_phred_counts);
    PyMem_Free(self->staging_end_anchored_base_counts);
    PyMem_Free(self->staging_end_anchored_phred_counts);
    PyMem_Free(self->base_counts);
    PyMem_Free(self->phred_counts);
    PyMem_Free(self->end_anchored_base_counts);
    PyMem_Free(self->end_anchored_phred_counts);
    PyTypeObject *tp = Py_TYPE((PyObject *)self);
    PyObject_Free(self);
    Py_DECREF(tp);
}

static PyObject *
QCMetrics__new__(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    Py_ssize_t end_anchor_length = DEFAULT_END_ANCHOR_LENGTH;
    static char *kwargnames[] = {"end_anchor_length", NULL};
    static char *format = "|n:QCMetrics";
    uint8_t phred_offset = 33;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, kwargnames,
                                     &end_anchor_length)) {
        return NULL;
    }
    if (end_anchor_length < 0 || end_anchor_length > UINT32_MAX) {
        PyErr_Format(PyExc_ValueError,
                     "end_anchor_length must be between 0 and %zd, got %zd",
                     (Py_ssize_t)UINT32_MAX, end_anchor_length);
        return NULL;
    }
    staging_base_table *staging_end_anchored_base_counts =
        PyMem_Calloc(end_anchor_length, sizeof(staging_base_table));
    staging_phred_table *staging_end_anchored_phred_counts =
        PyMem_Calloc(end_anchor_length, sizeof(staging_phred_table));
    base_table *end_anchored_base_counts =
        PyMem_Calloc(end_anchor_length, sizeof(base_table));
    phred_table *end_anchored_phred_counts =
        PyMem_Calloc(end_anchor_length, sizeof(phred_table));
    QCMetrics *self = PyObject_New(QCMetrics, type);
    if (self == NULL || staging_end_anchored_base_counts == NULL ||
        staging_end_anchored_phred_counts == NULL ||
        end_anchored_base_counts == NULL || end_anchored_phred_counts == NULL) {
        return PyErr_NoMemory();
    }
    self->phred_offset = phred_offset;
    self->staging_count = 0;
    self->end_anchor_length = end_anchor_length;
    self->max_length = 0;
    self->staging_base_counts = NULL;
    self->staging_phred_counts = NULL;
    self->staging_end_anchored_base_counts = staging_end_anchored_base_counts;
    self->staging_end_anchored_phred_counts = staging_end_anchored_phred_counts;
    self->base_counts = NULL;
    self->phred_counts = NULL;
    self->end_anchored_base_counts = end_anchored_base_counts;
    self->end_anchored_phred_counts = end_anchored_phred_counts;
    self->number_of_reads = 0;
    memset(self->gc_content, 0, 101 * sizeof(uint64_t));
    memset(self->phred_scores, 0, (PHRED_MAX + 1) * sizeof(uint64_t));
    return (PyObject *)self;
}

static int
QCMetrics_resize(QCMetrics *self, Py_ssize_t new_size)
{
    staging_base_table *staging_base_tmp = PyMem_Realloc(
        self->staging_base_counts, new_size * sizeof(staging_base_table));
    staging_phred_table *staging_phred_tmp = PyMem_Realloc(
        self->staging_phred_counts, new_size * sizeof(staging_phred_table));
    base_table *base_table_tmp =
        PyMem_Realloc(self->base_counts, new_size * sizeof(base_table));
    phred_table *phred_table_tmp =
        PyMem_Realloc(self->phred_counts, new_size * sizeof(phred_table));

    if (staging_base_tmp == NULL || staging_phred_tmp == NULL ||
        base_table_tmp == NULL || phred_table_tmp == NULL) {
        PyErr_NoMemory();
        PyMem_Free(staging_base_tmp);
        PyMem_Free(staging_phred_tmp);
        PyMem_Free(base_table_tmp);
        PyMem_Free(phred_table_tmp);
        return -1;
    }

    size_t old_size = self->max_length;
    size_t new_slots = new_size - old_size;
    memset(staging_base_tmp + old_size, 0, new_slots * sizeof(staging_base_table));
    memset(staging_phred_tmp + old_size, 0,
           new_slots * sizeof(staging_phred_table));
    memset(base_table_tmp + old_size, 0, new_slots * sizeof(base_table));
    memset(phred_table_tmp + old_size, 0, new_slots * sizeof(phred_table));

    self->staging_base_counts = staging_base_tmp;
    self->staging_phred_counts = staging_phred_tmp;
    self->base_counts = base_table_tmp;
    self->phred_counts = phred_table_tmp;
    self->max_length = new_size;
    return 0;
}

static void
QCMetrics_flush_staging(QCMetrics *self)
{
    if (self->staging_count == 0) {
        return;
    }
    uint64_t *base_counts = (uint64_t *)self->base_counts;
    uint16_t *staging_base_counts = (uint16_t *)self->staging_base_counts;
    size_t number_of_base_slots = self->max_length * NUC_TABLE_SIZE;
    /* base counts is only updated once every 65535 times. So make sure it
       does not pollute the cache and use non temporal prefetching. The
       same goes for phred counts.
    */
    non_temporal_write_prefetch(base_counts);
    for (size_t i = 0; i < number_of_base_slots; i++) {
        base_counts[i] += staging_base_counts[i];
        /* Fetch the next 64 byte cache line non-temporal. */
        non_temporal_write_prefetch(base_counts + i + 8);
    }
    memset(staging_base_counts, 0, number_of_base_slots * sizeof(uint16_t));

    uint64_t *phred_counts = (uint64_t *)self->phred_counts;
    uint16_t *staging_phred_counts = (uint16_t *)self->staging_phred_counts;
    size_t number_of_phred_slots = self->max_length * PHRED_TABLE_SIZE;
    non_temporal_write_prefetch(phred_counts);
    for (size_t i = 0; i < number_of_phred_slots; i++) {
        phred_counts[i] += staging_phred_counts[i];
        non_temporal_write_prefetch(phred_counts + i + 8);
    }
    memset(staging_phred_counts, 0, number_of_phred_slots * sizeof(uint16_t));

    size_t end_anchor_length = self->end_anchor_length;

    size_t end_anchor_base_slots = end_anchor_length * NUC_TABLE_SIZE;
    uint64_t *end_anchored_base_counts =
        (uint64_t *)self->end_anchored_base_counts;
    uint16_t *staging_end_anchored_base_counts =
        (uint16_t *)self->staging_end_anchored_base_counts;
    for (size_t i = 0; i < end_anchor_base_slots; i++) {
        end_anchored_base_counts[i] += staging_end_anchored_base_counts[i];
    }
    memset(staging_end_anchored_base_counts, 0,
           end_anchor_base_slots * sizeof(uint16_t));

    size_t end_anchor_phred_slots = end_anchor_length * PHRED_TABLE_SIZE;
    uint64_t *end_anchored_phred_counts =
        (uint64_t *)self->end_anchored_phred_counts;
    uint16_t *staging_end_anchored_phred_counts =
        (uint16_t *)self->staging_end_anchored_phred_counts;
    for (size_t i = 0; i < end_anchor_phred_slots; i++) {
        end_anchored_phred_counts[i] += staging_end_anchored_phred_counts[i];
    }
    memset(staging_end_anchored_phred_counts, 0,
           end_anchor_phred_slots * sizeof(uint16_t));
    self->staging_count = 0;
}

static inline int
QCMetrics_add_meta(QCMetrics *self, struct FastqMeta *meta)
{
    const uint8_t *record_start = meta->record_start;
    size_t sequence_length = meta->sequence_length;
    size_t full_end_anchor_length = self->end_anchor_length;
    size_t end_anchor_length = Py_MIN(full_end_anchor_length, sequence_length);
    size_t end_anchor_store_offset = full_end_anchor_length - end_anchor_length;

    const uint8_t *sequence = record_start + meta->sequence_offset;
    const uint8_t *qualities = record_start + meta->qualities_offset;

    if (sequence_length > self->max_length) {
        if (QCMetrics_resize(self, sequence_length) != 0) {
            return -1;
        }
    }

    self->number_of_reads += 1;
    if (self->staging_count >= UINT16_MAX) {
        QCMetrics_flush_staging(self);
    }
    self->staging_count += 1;

    staging_base_table *staging_base_counts_ptr = self->staging_base_counts;
    const uint8_t *sequence_ptr = sequence;
    const uint8_t *end_ptr = sequence + sequence_length;
    const uint8_t *unroll_end_ptr = end_ptr - 3;
    /* A 64-bit integer can be used as 2 consecutive 32 bit integers. Using
       a bit of shifting, this means no memory access is needed to count
       the nucleotide counts for the GC content calculation.
       We can also count at_counts and gc_counts together.  */
    static const uint64_t count_integers[5] = {
        /*  A   , C            , G            , T   , N */
        1ULL, 1ULL << 32ULL, 1ULL << 32ULL, 1ULL, 0};
    uint64_t base_counts0 = 0;
    uint64_t base_counts1 = 0;
    uint64_t base_counts2 = 0;
    uint64_t base_counts3 = 0;
    while (sequence_ptr < unroll_end_ptr) {
        uint64_t c0 = sequence_ptr[0];
        uint64_t c1 = sequence_ptr[1];
        uint64_t c2 = sequence_ptr[2];
        uint64_t c3 = sequence_ptr[3];
        uint64_t c0_index = NUCLEOTIDE_TO_INDEX[c0];
        uint64_t c1_index = NUCLEOTIDE_TO_INDEX[c1];
        uint64_t c2_index = NUCLEOTIDE_TO_INDEX[c2];
        uint64_t c3_index = NUCLEOTIDE_TO_INDEX[c3];
        base_counts0 += count_integers[c0_index];
        base_counts1 += count_integers[c1_index];
        base_counts2 += count_integers[c2_index];
        base_counts3 += count_integers[c3_index];
        staging_base_counts_ptr[0][c0_index] += 1;
        staging_base_counts_ptr[1][c1_index] += 1;
        staging_base_counts_ptr[2][c2_index] += 1;
        staging_base_counts_ptr[3][c3_index] += 1;
        sequence_ptr += 4;
        staging_base_counts_ptr += 4;
    }
    while (sequence_ptr < end_ptr) {
        uint64_t c = *sequence_ptr;
        uint64_t c_index = NUCLEOTIDE_TO_INDEX[c];
        base_counts0 += count_integers[c_index];
        staging_base_counts_ptr[0][c_index] += 1;
        sequence_ptr += 1;
        staging_base_counts_ptr += 1;
    }

    // End-anchored run while sequence still hot in cache
    staging_base_table *staging_end_anchored_bases =
        self->staging_end_anchored_base_counts + end_anchor_store_offset;
    sequence_ptr -= end_anchor_length;
    while (sequence_ptr < end_ptr) {
        size_t c = *sequence_ptr;
        size_t c_index = NUCLEOTIDE_TO_INDEX[c];
        staging_end_anchored_bases[0][c_index] += 1;
        staging_end_anchored_bases += 1;
        sequence_ptr += 1;
    }

    uint64_t base_counts =
        base_counts0 + base_counts1 + base_counts2 + base_counts3;
    uint64_t at_counts = base_counts & 0xFFFFFFFF;
    uint64_t gc_counts = (base_counts >> 32) & 0xFFFFFFFF;
    uint64_t total = gc_counts + at_counts;
    // if total == 0 there will be divide by 0, so only run if total > 0.
    if (total > 0) {
        double gc_content_percentage =
            (double)gc_counts * (double)100.0 / (double)total;
        uint64_t gc_content_index = (uint64_t)round(gc_content_percentage);
        assert(gc_content_index >= 0);
        assert(gc_content_index <= 100);
        self->gc_content[gc_content_index] += 1;
    }
    staging_phred_table *staging_phred_counts_ptr = self->staging_phred_counts;
    const uint8_t *qualities_ptr = qualities;
    const uint8_t *qualities_end_ptr = qualities + sequence_length;
    const uint8_t *qualities_unroll_end_ptr = qualities_end_ptr - 4;
    uint8_t phred_offset = self->phred_offset;
    double accumulator0 = 0.0;
    double accumulator1 = 0.0;
    double accumulator2 = 0.0;
    double accumulator3 = 0.0;
    while (qualities_ptr < qualities_unroll_end_ptr) {
        uint8_t q0 = qualities_ptr[0] - phred_offset;
        uint8_t q1 = qualities_ptr[1] - phred_offset;
        uint8_t q2 = qualities_ptr[2] - phred_offset;
        uint8_t q3 = qualities_ptr[3] - phred_offset;
        if (q0 > PHRED_MAX || q1 > PHRED_MAX || q2 > PHRED_MAX || q3 > PHRED_MAX) {
            break;
        }
        uint8_t q0_index = phred_to_index(q0);
        uint8_t q1_index = phred_to_index(q1);
        uint8_t q2_index = phred_to_index(q2);
        uint8_t q3_index = phred_to_index(q3);
        staging_phred_counts_ptr[0][q0_index] += 1;
        staging_phred_counts_ptr[1][q1_index] += 1;
        staging_phred_counts_ptr[2][q2_index] += 1;
        staging_phred_counts_ptr[3][q3_index] += 1;
        /* By writing it as multiple independent additions this takes advantage
           of out of order execution. While also making it obvious for the
           compiler that vectors can be used. */
        double error_rate0 = SCORE_TO_ERROR_RATE[q0];
        double error_rate1 = SCORE_TO_ERROR_RATE[q1];
        double error_rate2 = SCORE_TO_ERROR_RATE[q2];
        double error_rate3 = SCORE_TO_ERROR_RATE[q3];
        accumulator0 += error_rate0;
        accumulator1 += error_rate1;
        accumulator2 += error_rate2;
        accumulator3 += error_rate3;
        staging_phred_counts_ptr += 4;
        qualities_ptr += 4;
    }
    double accumulated_error_rate =
        accumulator0 + accumulator1 + accumulator2 + accumulator3;
    while (qualities_ptr < qualities_end_ptr) {
        uint8_t q = *qualities_ptr - phred_offset;
        if (q > PHRED_MAX) {
            PyErr_Format(PyExc_ValueError, "Not a valid phred character: %c",
                         *qualities_ptr);
            return -1;
        }
        uint8_t q_index = phred_to_index(q);
        staging_phred_counts_ptr[0][q_index] += 1;
        accumulated_error_rate += SCORE_TO_ERROR_RATE[q];
        staging_phred_counts_ptr += 1;
        qualities_ptr += 1;
    }

    // End-anchored run while qualities still hot in cache
    staging_phred_table *staging_end_anchored_phreds =
        self->staging_end_anchored_phred_counts + end_anchor_store_offset;
    qualities_ptr -= end_anchor_length;
    while (qualities_ptr < qualities_end_ptr) {
        size_t q = *qualities_ptr - phred_offset;
        size_t q_index = phred_to_index(q);
        staging_end_anchored_phreds[0][q_index] += 1;
        staging_end_anchored_phreds += 1;
        qualities_ptr += 1;
    }

    meta->accumulated_error_rate = accumulated_error_rate;
    if (sequence_length > 0) {
        double average_error_rate =
            accumulated_error_rate / (double)sequence_length;
        double average_phred = -10.0 * log10(average_error_rate);
        // Floor the average phred so q9.7 does not get represented as q10 but
        // q9. Otherwise the Q>=10 count is going to be off.
        uint64_t phred_score_index = (uint64_t)floor(average_phred);
        assert(phred_score_index >= 0);
        assert(phred_score_index <= PHRED_MAX);
        self->phred_scores[phred_score_index] += 1;
    }
    return 0;
}

PyDoc_STRVAR(QCMetrics_add_read__doc__,
             "add_read($self, read, /)\n"
             "--\n"
             "\n"
             "Add a read to the count metrics. \n"
             "\n"
             "  read\n"
             "    A FastqRecordView object.\n");

#define QCMetrics_add_read_method METH_O

static PyObject *
QCMetrics_add_read(QCMetrics *self, FastqRecordView *read)
{
    int is_view = is_FastqRecordView(self, read);
    if (is_view == -1) {
        return NULL;
    }
    else if (is_view == 0) {
        PyErr_Format(PyExc_TypeError,
                     "read should be a FastqRecordView object, got %R",
                     Py_TYPE((PyObject *)read));
        return NULL;
    }
    if (QCMetrics_add_meta(self, &read->meta) != 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(QCMetrics_add_record_array__doc__,
             "add_record_array($self, record_array, /)\n"
             "--\n"
             "\n"
             "Add a record_array to the count metrics. \n"
             "\n"
             "  record_array\n"
             "    A FastqRecordArrayView object.\n");

#define QCMetrics_add_record_array_method METH_O

static PyObject *
QCMetrics_add_record_array(QCMetrics *self, FastqRecordArrayView *record_array)
{
    int is_record_array = is_FastqRecordArrayView(self, record_array);
    if (is_record_array == -1) {
        return NULL;
    }
    else if (is_record_array == 0) {
        PyErr_Format(
            PyExc_TypeError,
            "record_array should be a FastqRecordArrayView object, got %R",
            Py_TYPE((PyObject *)record_array));
        return NULL;
    }
    Py_ssize_t number_of_records = Py_SIZE((PyObject *)record_array);
    struct FastqMeta *records = record_array->records;
    for (Py_ssize_t i = 0; i < number_of_records; i++) {
        if (QCMetrics_add_meta(self, records + i) != 0) {
            return NULL;
        }
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(QCMetrics_base_count_table__doc__,
             "base_count_table($self, /)\n"
             "--\n"
             "\n"
             "Return a array.array on the produced base count table. \n");

#define QCMetrics_base_count_table_method METH_NOARGS

static PyObject *
QCMetrics_base_count_table(QCMetrics *self, PyObject *Py_UNUSED(ignore))
{
    struct QCModuleState *state = get_qc_module_state_from_obj(self);
    if (state == NULL) {
        return NULL;
    }
    QCMetrics_flush_staging(self);
    return PythonArray_FromBuffer('Q', self->base_counts,
                                  self->max_length * sizeof(base_table),
                                  state->PythonArray_Type);
}

PyDoc_STRVAR(QCMetrics_phred_count_table__doc__,
             "phred_table($self, /)\n"
             "--\n"
             "\n"
             "Return a array.array on the produced phred count table. \n");

#define QCMetrics_phred_count_table_method METH_NOARGS

static PyObject *
QCMetrics_phred_count_table(QCMetrics *self, PyObject *Py_UNUSED(ignore))
{
    struct QCModuleState *state = get_qc_module_state_from_obj(self);
    if (state == NULL) {
        return NULL;
    }
    QCMetrics_flush_staging(self);
    return PythonArray_FromBuffer('Q', self->phred_counts,
                                  self->max_length * sizeof(phred_table),
                                  state->PythonArray_Type);
}

PyDoc_STRVAR(
    QCMetrics_end_anchored_base_count_table__doc__,
    "end_anchored_base_count_table($self, /)\n"
    "--\n"
    "\n"
    "Return a array.array on the produced end anchored base count table. \n");

#define QCMetrics_end_anchored_base_count_table_method METH_NOARGS

static PyObject *
QCMetrics_end_anchored_base_count_table(QCMetrics *self,
                                        PyObject *Py_UNUSED(ignore))
{
    struct QCModuleState *state = get_qc_module_state_from_obj(self);
    if (state == NULL) {
        return NULL;
    }
    QCMetrics_flush_staging(self);
    return PythonArray_FromBuffer('Q', self->end_anchored_base_counts,
                                  self->end_anchor_length * sizeof(base_table),
                                  state->PythonArray_Type);
}

PyDoc_STRVAR(
    QCMetrics_end_anchored_phred_count_table__doc__,
    "end_anchored_phred_table($self, /)\n"
    "--\n"
    "\n"
    "Return a array.array on the produced end anchored phred count table. \n");

#define QCMetrics_end_anchored_phred_count_table_method METH_NOARGS

static PyObject *
QCMetrics_end_anchored_phred_count_table(QCMetrics *self,
                                         PyObject *Py_UNUSED(ignore))
{
    struct QCModuleState *state = get_qc_module_state_from_obj(self);
    if (state == NULL) {
        return NULL;
    }
    QCMetrics_flush_staging(self);
    return PythonArray_FromBuffer('Q', self->end_anchored_phred_counts,
                                  self->end_anchor_length * sizeof(phred_table),
                                  state->PythonArray_Type);
}

PyDoc_STRVAR(QCMetrics_gc_content__doc__,
             "gc_content($self, /)\n"
             "--\n"
             "\n"
             "Return a array.array on the produced gc content counts. \n");

#define QCMetrics_gc_content_method METH_NOARGS

static PyObject *
QCMetrics_gc_content(QCMetrics *self, PyObject *Py_UNUSED(ignore))
{
    struct QCModuleState *state = get_qc_module_state_from_obj(self);
    if (state == NULL) {
        return NULL;
    }
    QCMetrics_flush_staging(self);
    return PythonArray_FromBuffer(
        'Q', self->gc_content, sizeof(self->gc_content), state->PythonArray_Type);
}

PyDoc_STRVAR(
    QCMetrics_phred_scores__doc__,
    "phred_scores($self, /)\n"
    "--\n"
    "\n"
    "Return a array.array on the produced average phred score counts. \n");

#define QCMetrics_phred_scores_method METH_NOARGS

static PyObject *
QCMetrics_phred_scores(QCMetrics *self, PyObject *Py_UNUSED(ignore))
{
    struct QCModuleState *state = get_qc_module_state_from_obj(self);
    if (state == NULL) {
        return NULL;
    }
    QCMetrics_flush_staging(self);
    return PythonArray_FromBuffer('Q', self->phred_scores,
                                  sizeof(self->phred_scores),
                                  state->PythonArray_Type);
}

static PyMethodDef QCMetrics_methods[] = {
    {"add_read", (PyCFunction)QCMetrics_add_read, QCMetrics_add_read_method,
     QCMetrics_add_read__doc__},
    {"add_record_array", (PyCFunction)QCMetrics_add_record_array,
     QCMetrics_add_record_array_method, QCMetrics_add_record_array__doc__},
    {"base_count_table", (PyCFunction)QCMetrics_base_count_table,
     QCMetrics_base_count_table_method, QCMetrics_base_count_table__doc__},
    {"phred_count_table", (PyCFunction)QCMetrics_phred_count_table,
     QCMetrics_phred_count_table_method, QCMetrics_phred_count_table__doc__},
    {"end_anchored_base_count_table",
     (PyCFunction)QCMetrics_end_anchored_base_count_table,
     QCMetrics_end_anchored_base_count_table_method,
     QCMetrics_end_anchored_base_count_table__doc__},
    {"end_anchored_phred_count_table",
     (PyCFunction)QCMetrics_end_anchored_phred_count_table,
     QCMetrics_end_anchored_phred_count_table_method,
     QCMetrics_end_anchored_phred_count_table__doc__},

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
    {"end_anchor_length", T_ULONGLONG, offsetof(QCMetrics, end_anchor_length),
     READONLY, "The length of the end of the read that is sampled."},
    {NULL},
};

static PyType_Slot QCMetrics_slots[] = {
    {Py_tp_dealloc, (destructor)QCMetrics_dealloc},
    {Py_tp_new, QCMetrics__new__},
    {Py_tp_members, QCMetrics_members},
    {Py_tp_methods, QCMetrics_methods},
    {0, NULL},
};

static PyType_Spec QCMetrics_spec = {
    .name = "_qc.QCMetrics",
    .basicsize = sizeof(QCMetrics),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT,
    .slots = QCMetrics_slots,
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

struct AdapterCounts {
    uint64_t *forward;
    uint64_t *reverse;
};

typedef struct AdapterCounterStruct {
    PyObject_HEAD
    size_t number_of_adapters;
    size_t max_length;
    size_t number_of_sequences;
    struct AdapterCounts *adapter_counter;
    PyObject *adapters;
    size_t number_of_matchers;
    bitmask_t *init_masks;
    bitmask_t *found_masks;
    bitmask_t (*bitmasks)[NUC_TABLE_SIZE];
    /* Same as bitmasks, but better organization for vectorized approach. */
    bitmask_t (*by_four_bitmasks)[NUC_TABLE_SIZE][4];
    AdapterSequence **adapter_sequences;
} AdapterCounter;

static void
AdapterCounter_dealloc(AdapterCounter *self)
{
    Py_XDECREF(self->adapters);
    if (self->adapter_counter != NULL) {
        for (size_t i = 0; i < self->number_of_adapters; i++) {
            struct AdapterCounts counts = self->adapter_counter[i];
            PyMem_Free(counts.forward);
            PyMem_Free(counts.reverse);
        }
    }
    PyMem_Free(self->adapter_counter);
    if (self->adapter_sequences != NULL) {
        for (size_t i = 0; i < self->number_of_matchers; i++) {
            PyMem_Free(self->adapter_sequences[i]);
        }
    }
    PyMem_Free(self->found_masks);
    PyMem_Free(self->init_masks);
    PyMem_Free(self->bitmasks);
    PyMem_Free(self->by_four_bitmasks);
    PyMem_Free(self->adapter_sequences);

    PyTypeObject *tp = Py_TYPE((PyObject *)self);
    PyObject_Free(self);
    Py_XDECREF((PyObject *)tp);
}

static void
populate_bitmask(bitmask_t bitmask[NUC_TABLE_SIZE], char *word, size_t word_length)
{
    for (size_t i = 0; i < word_length; i++) {
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
    size_t number_of_adapters = PyTuple_Size(adapters);
    size_t number_of_matchers = 1;
    size_t matcher_length = 0;
    if (number_of_adapters < 1) {
        PyErr_SetString(PyExc_ValueError, "At least one adapter is expected");
        goto error;
    }
    for (size_t i = 0; i < number_of_adapters; i++) {
        PyObject *adapter = PyTuple_GetItem(adapters, i);
        if (!PyUnicode_CheckExact(adapter)) {
            PyErr_Format(PyExc_TypeError,
                         "All adapter sequences must be of type str, "
                         "got %R, for %R",
                         Py_TYPE((PyObject *)adapter), adapter);
            goto error;
        }
        Py_ssize_t utf8size = 0;
        Py_ssize_t adapter_length = PyUnicode_GetLength(adapter);
        PyUnicode_AsUTF8AndSize(adapter, &utf8size);
        if (adapter_length != utf8size) {
            PyErr_Format(PyExc_ValueError,
                         "Adapter must contain only ASCII characters: %R",
                         adapter);
            goto error;
        }
        if ((size_t)adapter_length > MAX_SEQUENCE_SIZE) {
            PyErr_Format(PyExc_ValueError,
                         "Maximum adapter size is %d, got %zd for %R",
                         MAX_SEQUENCE_SIZE, adapter_length, adapter);
            goto error;
        }
        if (matcher_length + adapter_length > MAX_SEQUENCE_SIZE) {
            matcher_length = adapter_length;
            number_of_matchers += 1;
        }
        else {
            matcher_length += adapter_length;
        }
    }
    self = PyObject_New(AdapterCounter, type);
    self->adapter_counter =
        PyMem_Calloc(number_of_adapters, sizeof(struct AdapterCounts));
    /* Ensure there is enough space to always do vector loads. */
    size_t matcher_array_size = number_of_matchers + 3;
    self->found_masks = PyMem_Calloc(matcher_array_size, sizeof(bitmask_t));
    self->init_masks = PyMem_Calloc(matcher_array_size, sizeof(bitmask_t));
    self->adapter_sequences =
        PyMem_Calloc(matcher_array_size, sizeof(AdapterSequence *));
    self->bitmasks =
        PyMem_Calloc(matcher_array_size, NUC_TABLE_SIZE * sizeof(bitmask_t));
    self->by_four_bitmasks = PyMem_Calloc(
        matcher_array_size / 4, NUC_TABLE_SIZE * 4 * sizeof(bitmask_t));
    if (self->adapter_counter == NULL || self->found_masks == NULL ||
        self->init_masks == NULL || self->adapter_sequences == NULL ||
        self->bitmasks == NULL || self->by_four_bitmasks == NULL) {
        PyErr_NoMemory();
        goto error;
    }
    self->max_length = 0;
    self->number_of_adapters = number_of_adapters;
    self->number_of_matchers = number_of_matchers;
    self->number_of_sequences = 0;
    size_t adapter_index = 0;
    size_t matcher_index = 0;
    PyObject *adapter;
    Py_ssize_t adapter_length;
    char machine_word[MACHINE_WORD_BITS];
    matcher_index = 0;
    while (adapter_index < number_of_adapters) {
        bitmask_t found_mask = 0;
        bitmask_t init_mask = 0;
        size_t adapter_in_word_index = 0;
        size_t word_index = 0;
        memset(machine_word, 0, MACHINE_WORD_BITS);
        while (adapter_index < number_of_adapters) {
            adapter = PyTuple_GetItem(adapters, adapter_index);
            const char *adapter_data =
                PyUnicode_AsUTF8AndSize(adapter, &adapter_length);
            if ((word_index + adapter_length) > MACHINE_WORD_BITS) {
                break;
            }
            memcpy(machine_word + word_index, adapter_data, adapter_length);
            init_mask |= (1ULL << word_index);
            word_index += adapter_length;
            AdapterSequence adapter_sequence = {
                .adapter_index = adapter_index,
                .adapter_length = adapter_length,
                .found_mask = 1ULL << (word_index - 1), /* Last character */
            };
            AdapterSequence empty_adapter_sequence = {0, 0, 0};
            AdapterSequence *adapt_tmp = PyMem_Realloc(
                self->adapter_sequences[matcher_index],
                (adapter_in_word_index + 2) * sizeof(AdapterSequence));
            if (adapt_tmp == NULL) {
                PyErr_NoMemory();
                goto error;
            }
            self->adapter_sequences[matcher_index] = adapt_tmp;
            self->adapter_sequences[matcher_index][adapter_in_word_index] =
                adapter_sequence;
            self->adapter_sequences[matcher_index][adapter_in_word_index + 1] =
                empty_adapter_sequence;
            found_mask |= adapter_sequence.found_mask;
            adapter_in_word_index += 1;
            adapter_index += 1;
        }
        populate_bitmask(self->bitmasks[matcher_index], machine_word, word_index);
        self->found_masks[matcher_index] = found_mask;
        self->init_masks[matcher_index] = init_mask;
        matcher_index += 1;
    }
    /* Initialize an array for better vectorized loading. Doing it here is
       much more efficient than doing it for every string. */
    for (size_t i = 0; i < self->number_of_matchers; i += 4) {
        for (size_t j = 0; j < NUC_TABLE_SIZE; j++) {
            self->by_four_bitmasks[i / 4][j][0] = self->bitmasks[i][j];
            self->by_four_bitmasks[i / 4][j][1] = self->bitmasks[i + 1][j];
            self->by_four_bitmasks[i / 4][j][2] = self->bitmasks[i + 2][j];
            self->by_four_bitmasks[i / 4][j][3] = self->bitmasks[i + 3][j];
        }
    }
    self->adapters = adapters;
    return (PyObject *)self;

error:
    Py_XDECREF(adapters);
    Py_XDECREF((PyObject *)self);
    return NULL;
}

static int
AdapterCounter_resize(AdapterCounter *self, size_t new_size)
{
    if (self->max_length >= new_size) {
        return 0;
    }
    size_t old_size = self->max_length;
    for (size_t i = 0; i < self->number_of_adapters; i++) {
        struct AdapterCounts counts = self->adapter_counter[i];
        uint64_t *tmp_forward =
            PyMem_Realloc(counts.forward, new_size * sizeof(uint64_t));
        if (tmp_forward == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        memset(tmp_forward + old_size, 0,
               (new_size - old_size) * sizeof(uint64_t));
        self->adapter_counter[i].forward = tmp_forward;
        uint64_t *tmp_reverse =
            PyMem_Realloc(counts.reverse, new_size * sizeof(uint64_t));
        if (tmp_reverse == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        memset(tmp_reverse + old_size, 0,
               (new_size - old_size) * sizeof(uint64_t));
        self->adapter_counter[i].reverse = tmp_reverse;
    }
    self->max_length = new_size;
    return 0;
}

static inline bitmask_t
update_adapter_count_array(size_t position, size_t length, bitmask_t match,
                           bitmask_t already_found,
                           AdapterSequence *adapter_sequences,
                           struct AdapterCounts *adapter_counter)
{
    size_t adapter_index = 0;
    while (true) {
        AdapterSequence *adapter = adapter_sequences + adapter_index;
        size_t adapter_length = adapter->adapter_length;
        if (adapter_length == 0) {
            break;
        }
        bitmask_t adapter_found_mask = adapter->found_mask;
        if (adapter_found_mask & already_found) {
            adapter_index += 1;
            continue;
        }
        if (match & adapter_found_mask) {
            size_t found_position = position - adapter_length + 1;
            struct AdapterCounts *counts =
                adapter_counter + adapter->adapter_index;
            counts->forward[found_position] += 1;
            counts->reverse[(length - 1) - found_position] += 1;
            already_found |= adapter_found_mask;
        }
        adapter_index += 1;
    }
    return already_found;
}

static void
find_single_matcher(const uint8_t *sequence, size_t sequence_length,
                    const bitmask_t *restrict init_masks,
                    const bitmask_t *restrict found_masks,
                    const bitmask_t (*bitmasks)[NUC_TABLE_SIZE],
                    AdapterSequence **adapter_sequences_store,
                    struct AdapterCounts *adapter_counter)
{
    bitmask_t found_mask = found_masks[0];
    bitmask_t init_mask = init_masks[0];
    bitmask_t R = 0;
    bitmask_t already_found = 0;
    const bitmask_t *bitmask = bitmasks[0];
    AdapterSequence *adapter_sequences = adapter_sequences_store[0];
    for (size_t pos = 0; pos < sequence_length; pos++) {
        R <<= 1;
        R |= init_mask;
        uint8_t index = NUCLEOTIDE_TO_INDEX[sequence[pos]];
        R &= bitmask[index];
        if (R & found_mask) {
            already_found = update_adapter_count_array(
                pos, sequence_length, R, already_found, adapter_sequences,
                adapter_counter);
        }
    }
}

static void (*find_four_matchers)(const uint8_t *sequence, size_t sequence_length,
                                  const bitmask_t *restrict init_masks,
                                  const bitmask_t *restrict found_masks,
                                  const bitmask_t (*by_four_bitmasks)[4],
                                  AdapterSequence **adapter_sequences_store,
                                  struct AdapterCounts *adapter_counter) = NULL;

#if COMPILER_HAS_TARGETED_DISPATCH && BUILD_IS_X86_64
__attribute__((__target__("avx2"))) static void
find_four_matchers_avx2(const uint8_t *sequence, size_t sequence_length,
                        const bitmask_t *restrict init_masks,
                        const bitmask_t *restrict found_masks,
                        const bitmask_t (*by_four_bitmasks)[4],
                        AdapterSequence **adapter_sequences_store,
                        struct AdapterCounts *adapter_counter)
{
    bitmask_t fmask0 = found_masks[0];
    bitmask_t fmask1 = found_masks[1];
    bitmask_t fmask2 = found_masks[2];
    bitmask_t fmask3 = found_masks[3];
    bitmask_t already_found0 = 0;
    bitmask_t already_found1 = 0;
    bitmask_t already_found2 = 0;
    bitmask_t already_found3 = 0;

    __m256i found_mask = _mm256_loadu_si256((const __m256i *)(found_masks));
    __m256i init_mask = _mm256_loadu_si256((const __m256i *)(init_masks));

    __m256i R = _mm256_setzero_si256();
    const bitmask_t(*bitmask)[4] = by_four_bitmasks;

    for (size_t pos = 0; pos < sequence_length; pos++) {
        R = _mm256_slli_epi64(R, 1);
        R = _mm256_or_si256(R, init_mask);
        uint8_t index = NUCLEOTIDE_TO_INDEX[sequence[pos]];
        R = _mm256_and_si256(R, _mm256_loadu_si256((__m256i *)bitmask[index]));

        __m256i check = _mm256_and_si256(R, found_mask);
        /* Adding 0b01111111 (127) to any number higher than 0 sets the bit for
           128. Movemask collects these bits. This way we can test if there is
           a 1 across the entire 256-bit vector. */
        int check_int =
            _mm256_movemask_epi8(_mm256_adds_epu8(check, _mm256_set1_epi8(127)));
        if (check_int) {
            bitmask_t Rray[4];
            _mm256_storeu_si256(((__m256i *)Rray), R);

            if (Rray[0] & fmask0) {
                already_found0 = update_adapter_count_array(
                    pos, sequence_length, Rray[0], already_found0,
                    adapter_sequences_store[0], adapter_counter);
            }
            if (Rray[1] & fmask1) {
                already_found1 = update_adapter_count_array(
                    pos, sequence_length, Rray[1], already_found1,
                    adapter_sequences_store[1], adapter_counter);
            }
            if (Rray[2] & fmask2) {
                already_found2 = update_adapter_count_array(
                    pos, sequence_length, Rray[2], already_found2,
                    adapter_sequences_store[2], adapter_counter);
            }
            if (Rray[3] & fmask3) {
                already_found3 = update_adapter_count_array(
                    pos, sequence_length, Rray[3], already_found3,
                    adapter_sequences_store[3], adapter_counter);
            }
        }
    }
}

/* Constructor runs at dynamic load time */
__attribute__((constructor)) static void
find_four_matchers_init_func_ptr(void)
{
    if (__builtin_cpu_supports("avx2")) {
        find_four_matchers = find_four_matchers_avx2;
    }
    else {
        find_four_matchers = NULL;
    }
}
#endif

static int
AdapterCounter_add_meta(AdapterCounter *self, struct FastqMeta *meta)
{
    self->number_of_sequences += 1;
    uint8_t *sequence = meta->record_start + meta->sequence_offset;
    size_t sequence_length = meta->sequence_length;

    if (sequence_length > self->max_length) {
        int ret = AdapterCounter_resize(self, sequence_length);
        if (ret != 0) {
            return -1;
        }
    }
    size_t number_of_matchers = self->number_of_matchers;
    size_t matcher_index = 0;
    bitmask_t *found_masks = self->found_masks;
    bitmask_t *init_masks = self->init_masks;
    bitmask_t(*bitmasks)[5] = self->bitmasks;
    AdapterSequence **adapter_sequences = self->adapter_sequences;
    struct AdapterCounts *adapter_count_array = self->adapter_counter;
    while (matcher_index < number_of_matchers) {
        /* Only run when a vectorized function pointer is initialized */
        if (find_four_matchers && number_of_matchers - matcher_index > 1) {
            find_four_matchers(
                sequence, sequence_length, init_masks + matcher_index,
                found_masks + matcher_index,
                self->by_four_bitmasks[matcher_index / 4],
                adapter_sequences + matcher_index, adapter_count_array);
            matcher_index += 4;
            continue;
        }
        find_single_matcher(
            sequence, sequence_length, init_masks + matcher_index,
            found_masks + matcher_index, bitmasks + matcher_index,
            adapter_sequences + matcher_index, adapter_count_array);
        matcher_index += 1;
    }
    return 0;
}

PyDoc_STRVAR(AdapterCounter_add_read__doc__,
             "add_read($self, read, /)\n"
             "--\n"
             "\n"
             "Add a read to the adapter counter. \n"
             "\n"
             "  read\n"
             "    A FastqRecordView object.\n");

#define AdapterCounter_add_read_method METH_O

static PyObject *
AdapterCounter_add_read(AdapterCounter *self, FastqRecordView *read)
{
    int is_view = is_FastqRecordView(self, read);
    if (is_view == -1) {
        return NULL;
    }
    else if (is_view == 0) {
        PyErr_Format(PyExc_TypeError,
                     "read should be a FastqRecordView object, got %R",
                     Py_TYPE((PyObject *)read));
        return NULL;
    }
    if (AdapterCounter_add_meta(self, &read->meta) != 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(AdapterCounter_add_record_array__doc__,
             "add_record_array($self, record_array, /)\n"
             "--\n"
             "\n"
             "Add a record_array to the adapter counter. \n"
             "\n"
             "  record_array\n"
             "    A FastqRecordArrayView object.\n");

#define AdapterCounter_add_record_array_method METH_O

static PyObject *
AdapterCounter_add_record_array(AdapterCounter *self,
                                FastqRecordArrayView *record_array)
{
    int is_record_array = is_FastqRecordArrayView(self, record_array);
    if (is_record_array == -1) {
        return NULL;
    }
    else if (is_record_array == 0) {
        PyErr_Format(
            PyExc_TypeError,
            "record_array should be a FastqRecordArrayView object, got %R",
            Py_TYPE((PyObject *)record_array));
        return NULL;
    }
    Py_ssize_t number_of_records = Py_SIZE((PyObject *)record_array);
    struct FastqMeta *records = record_array->records;
    for (Py_ssize_t i = 0; i < number_of_records; i++) {
        if (AdapterCounter_add_meta(self, records + i) != 0) {
            return NULL;
        }
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(AdapterCounter_get_counts__doc__,
             "get_counts($self, /)\n"
             "--\n"
             "\n"
             "Return the counts as a list of tuples. Each tuple contains the "
             "adapter, \n"
             "and an array.array counts per position. \n");

#define AdapterCounter_get_counts_method METH_NOARGS

static PyObject *
AdapterCounter_get_counts(AdapterCounter *self, PyObject *Py_UNUSED(ignore))
{
    struct QCModuleState *state = get_qc_module_state_from_obj(self);
    PyTypeObject *PythonArray_Type = state->PythonArray_Type;
    PyObject *counts_list = PyList_New(self->number_of_adapters);
    if (counts_list == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    for (size_t i = 0; i < self->number_of_adapters; i++) {
        PyObject *counts_forward = PythonArray_FromBuffer(
            'Q', self->adapter_counter[i].forward,
            self->max_length * sizeof(uint64_t), PythonArray_Type);
        if (counts_forward == NULL) {
            return NULL;
        }
        PyObject *counts_reverse = PythonArray_FromBuffer(
            'Q', self->adapter_counter[i].reverse,
            self->max_length * sizeof(uint64_t), PythonArray_Type);
        if (counts_reverse == NULL) {
            return NULL;
        }
        PyObject *adapter = PyTuple_GetItem(self->adapters, i);
        Py_INCREF(adapter);
        PyObject *tup = PyTuple_New(3);
        PyTuple_SetItem(tup, 0, adapter);
        PyTuple_SetItem(tup, 1, counts_forward);
        PyTuple_SetItem(tup, 2, counts_reverse);
        PyList_SetItem(counts_list, i, tup);
    }
    return counts_list;
}

static PyMethodDef AdapterCounter_methods[] = {
    {"add_read", (PyCFunction)AdapterCounter_add_read,
     AdapterCounter_add_read_method, AdapterCounter_add_read__doc__},
    {"add_record_array", (PyCFunction)AdapterCounter_add_record_array,
     AdapterCounter_add_record_array_method,
     AdapterCounter_add_record_array__doc__},
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

static PyType_Slot AdapterCounter_slots[] = {
    {Py_tp_dealloc, (destructor)AdapterCounter_dealloc},
    {Py_tp_new, (newfunc)AdapterCounter__new__},
    {Py_tp_members, AdapterCounter_members},
    {Py_tp_methods, AdapterCounter_methods},
    {0, NULL},
};

static PyType_Spec AdapterCounter_spec = {
    .name = "_qc.AdapterCounter",
    .basicsize = sizeof(AdapterCounter),
    .flags = Py_TPFLAGS_DEFAULT,
    .slots = AdapterCounter_slots,
};

/********************
 * Per Tile Quality *
 ********************/

typedef struct _TileQualityStruct {
    uint64_t *length_counts;
    double *total_errors;
} TileQuality;

typedef struct _PerTileQualityStruct {
    PyObject_HEAD
    uint8_t phred_offset;
    char skipped;
    TileQuality *tile_qualities;
    size_t number_of_tiles;
    size_t max_length;
    size_t number_of_reads;
    PyObject *skipped_reason;
} PerTileQuality;

static void
PerTileQuality_dealloc(PerTileQuality *self)
{
    Py_XDECREF(self->skipped_reason);
    for (size_t i = 0; i < self->number_of_tiles; i++) {
        TileQuality tile_qual = self->tile_qualities[i];
        PyMem_Free(tile_qual.length_counts);
        PyMem_Free(tile_qual.total_errors);
    }
    PyMem_Free(self->tile_qualities);
    PyTypeObject *tp = Py_TYPE((PyObject *)self);
    PyObject_Free(self);
    Py_XDECREF((PyObject *)tp);
}

static PyObject *
PerTileQuality__new__(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    static char *kwargnames[] = {NULL};
    static char *format = ":PerTileQuality";
    uint8_t phred_offset = 33;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, kwargnames)) {
        return NULL;
    }
    PerTileQuality *self = PyObject_New(PerTileQuality, type);
    self->max_length = 0;
    self->phred_offset = phred_offset;
    self->tile_qualities = NULL;
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
    TileQuality *new_qualities =
        PyMem_Realloc(self->tile_qualities, highest_tile * sizeof(TileQuality));
    if (new_qualities == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    size_t previous_number_of_tiles = self->number_of_tiles;
    memset(new_qualities + previous_number_of_tiles, 0,
           (highest_tile - previous_number_of_tiles) * sizeof(TileQuality));
    self->tile_qualities = new_qualities;
    self->number_of_tiles = highest_tile;
    return 0;
}

static int
PerTileQuality_resize_tiles(PerTileQuality *self, size_t new_length)
{
    if (new_length < self->max_length) {
        return 0;
    }
    TileQuality *tile_qualities = self->tile_qualities;
    size_t number_of_tiles = self->number_of_tiles;
    size_t old_length = self->max_length;
    for (size_t i = 0; i < number_of_tiles; i++) {
        TileQuality *tile_quality = tile_qualities + i;
        if (tile_quality->length_counts == NULL &&
            tile_quality->total_errors == NULL) {
            continue;
        }
        uint64_t *length_counts = PyMem_Realloc(tile_quality->length_counts,
                                                new_length * sizeof(uint64_t));
        double *total_errors = PyMem_Realloc(tile_quality->total_errors,
                                             new_length * sizeof(double));

        if (length_counts == NULL || total_errors == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        memset(length_counts + old_length, 0,
               (new_length - old_length) * sizeof(uint64_t));
        memset(total_errors + old_length, 0,
               (new_length - old_length) * sizeof(double));
        tile_quality->length_counts = length_counts;
        tile_quality->total_errors = total_errors;
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
static Py_ssize_t
illumina_header_to_tile_id(const uint8_t *header, size_t header_length)
{
    /* The following link contains the header format:
       https://support.illumina.com/help/BaseSpace_OLH_009008/Content/Source/Informatics/BS/FileFormat_FASTQ-files_swBS.htm
       It reports the following format:
       @<instrument>:<run number>:<flowcell
       ID>:<lane>:<tile>:<x-pos>:<y-pos>:<UMI> <read>:<is filtered>:<control
       number>:<index> The tile ID is after the fourth colon.
    */
    const uint8_t *header_end = header + header_length;
    const uint8_t *cursor = header;
    size_t cursor_count = 0;
    while (cursor < header_end) {
        if (*cursor == ':') {
            cursor_count += 1;
            if (cursor_count == 4) {
                break;
            }
        }
        cursor += 1;
    }
    cursor += 1;
    const uint8_t *tile_start = cursor;
    while (cursor < header_end) {
        if (*cursor == ':') {
            const uint8_t *tile_end = cursor;
            size_t tile_length = tile_end - tile_start;
            return unsigned_decimal_integer_from_string(tile_start, tile_length);
        }
        cursor += 1;
    }
    return -1;
}

static int
PerTileQuality_add_meta(PerTileQuality *self, struct FastqMeta *meta)
{
    if (self->skipped) {
        return 0;
    }
    uint8_t *record_start = meta->record_start;
    const uint8_t *header = record_start;
    size_t header_length = meta->name_length;
    const uint8_t *qualities = record_start + meta->qualities_offset;
    size_t sequence_length = meta->sequence_length;
    uint8_t phred_offset = self->phred_offset;

    Py_ssize_t tile_id = illumina_header_to_tile_id(header, header_length);
    if (tile_id == -1) {
        PyObject *header_obj =
            PyUnicode_DecodeASCII((const char *)header, header_length, NULL);
        if (header_obj == NULL) {
            return -1;
        }
        self->skipped_reason =
            PyUnicode_FromFormat("Can not parse header: %R", header_obj);
        Py_DECREF(header_obj);
        self->skipped = 1;
        return 0;
    }

    if (sequence_length > self->max_length) {
        if (PerTileQuality_resize_tiles(self, sequence_length) != 0) {
            return -1;
        }
    }

    /* Tile index must be one less than the highest number of tiles otherwise
       the index is not in the tile array. */
    if (((size_t)tile_id + 1) > self->number_of_tiles) {
        if (PerTileQuality_resize_tile_array(self, tile_id + 1) != 0) {
            return -1;
        }
    }

    TileQuality *tile_quality = self->tile_qualities + tile_id;
    if (tile_quality->length_counts == NULL && tile_quality->total_errors == NULL) {
        uint64_t *length_counts =
            PyMem_Malloc(self->max_length * sizeof(uint64_t));
        double *total_errors = PyMem_Malloc(self->max_length * sizeof(double));
        if (length_counts == NULL || total_errors == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        memset(length_counts, 0, self->max_length * sizeof(uint64_t));
        memset(total_errors, 0, self->max_length * sizeof(double));
        tile_quality->length_counts = length_counts;
        tile_quality->total_errors = total_errors;
    }

    self->number_of_reads += 1;
    if (sequence_length == 0) {
        return 0;
    }
    tile_quality->length_counts[sequence_length - 1] += 1;
    double *restrict total_errors = tile_quality->total_errors;
    double *restrict error_cursor = total_errors;
    const uint8_t *qualities_end = qualities + sequence_length;
    const uint8_t *restrict qualities_ptr = qualities;
    const uint8_t *qualities_unroll_end = qualities_end - 3;
    while (qualities_ptr < qualities_unroll_end) {
        uint8_t phred0 = qualities_ptr[0] - phred_offset;
        uint8_t phred1 = qualities_ptr[1] - phred_offset;
        uint8_t phred2 = qualities_ptr[2] - phred_offset;
        uint8_t phred3 = qualities_ptr[3] - phred_offset;
        if (phred0 > PHRED_MAX || phred1 > PHRED_MAX || phred2 > PHRED_MAX ||
            phred3 > PHRED_MAX) {
            // Let the scalar loop handle the error
            break;
        }
        double error0 = SCORE_TO_ERROR_RATE[phred0];
        double error1 = SCORE_TO_ERROR_RATE[phred1];
        double error2 = SCORE_TO_ERROR_RATE[phred2];
        double error3 = SCORE_TO_ERROR_RATE[phred3];
        error_cursor[0] += error0;
        error_cursor[1] += error1;
        error_cursor[2] += error2;
        error_cursor[3] += error3;
        qualities_ptr += 4;
        error_cursor += 4;
    }
    while (qualities_ptr < qualities_end) {
        uint8_t q = *qualities_ptr - phred_offset;
        if (q > PHRED_MAX) {
            PyErr_Format(PyExc_ValueError, "Not a valid phred character: %c",
                         *qualities_ptr);
            return -1;
        }
        *error_cursor += SCORE_TO_ERROR_RATE[q];
        qualities_ptr += 1;
        error_cursor += 1;
    }
    return 0;
}

PyDoc_STRVAR(PerTileQuality_add_read__doc__,
             "add_read($self, read, /)\n"
             "--\n"
             "\n"
             "Add a read to the PerTileQuality Metrics. \n"
             "\n"
             "  read\n"
             "    A FastqRecordView object.\n");

#define PerTileQuality_add_read_method METH_O

static PyObject *
PerTileQuality_add_read(PerTileQuality *self, FastqRecordView *read)
{
    if (self->skipped) {
        Py_RETURN_NONE;
    }
    int is_view = is_FastqRecordView(self, read);
    if (is_view == -1) {
        return NULL;
    }
    else if (is_view == 0) {
        PyErr_Format(PyExc_TypeError,
                     "read should be a FastqRecordView object, got %R",
                     Py_TYPE((PyObject *)read));
        return NULL;
    }
    if (PerTileQuality_add_meta(self, &read->meta) != 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(PerTileQuality_add_record_array__doc__,
             "add_record_array($self, record_array, /)\n"
             "--\n"
             "\n"
             "Add a record_array to the PerTileQuality metrics. \n"
             "\n"
             "  record_array\n"
             "    A FastqRecordArrayView object.\n");

#define PerTileQuality_add_record_array_method METH_O

static PyObject *
PerTileQuality_add_record_array(PerTileQuality *self,
                                FastqRecordArrayView *record_array)
{
    if (self->skipped) {
        Py_RETURN_NONE;
    }
    int is_record_array = is_FastqRecordArrayView(self, record_array);
    if (is_record_array == -1) {
        return NULL;
    }
    else if (is_record_array == 0) {
        PyErr_Format(
            PyExc_TypeError,
            "record_array should be a FastqRecordArrayView object, got %R",
            Py_TYPE((PyObject *)record_array));
        return NULL;
    }
    Py_ssize_t number_of_records = Py_SIZE((PyObject *)record_array);
    struct FastqMeta *records = record_array->records;
    for (Py_ssize_t i = 0; i < number_of_records; i++) {
        if (PerTileQuality_add_meta(self, records + i) != 0) {
            return NULL;
        }
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(PerTileQuality_get_tile_counts__doc__,
             "get_tile_counts($self, /)\n"
             "--\n"
             "\n"
             "Get a list of tuples with the tile IDs and a list of their "
             "summed errors and\n"
             "a list of their counts. \n");

#define PerTileQuality_get_tile_counts_method METH_NOARGS

static PyObject *
PerTileQuality_get_tile_counts(PerTileQuality *self, PyObject *Py_UNUSED(ignore))
{
    TileQuality *tile_qualities = self->tile_qualities;
    size_t maximum_tile = self->number_of_tiles;
    size_t tile_length = self->max_length;
    PyObject *result = PyList_New(0);
    if (result == NULL) {
        return PyErr_NoMemory();
    }

    for (size_t i = 0; i < maximum_tile; i++) {
        TileQuality *tile_quality = tile_qualities + i;
        double *total_errors = tile_quality->total_errors;
        uint64_t *length_counts = tile_quality->length_counts;
        if (length_counts == NULL && total_errors == NULL) {
            continue;
        }
        PyObject *entry = PyTuple_New(3);
        PyObject *tile_id = PyLong_FromSize_t(i);
        PyObject *summed_error_list = PyList_New(tile_length);
        PyObject *count_list = PyList_New(tile_length);
        if (entry == NULL || tile_id == NULL || summed_error_list == NULL ||
            count_list == NULL) {
            Py_DECREF(result);
            return PyErr_NoMemory();
        }
        /* Work back from the lenght counts. If we have 200 reads total and a
           100 are length 150 and a 100 are length 120. This means we have
           a 100 bases at each position 120-150 and 200 bases at 0-120. */
        uint64_t total_bases = 0;
        for (Py_ssize_t j = tile_length - 1; j >= 0; j -= 1) {
            total_bases += length_counts[j];
            PyObject *summed_error_obj = PyFloat_FromDouble(total_errors[j]);
            PyObject *count_obj = PyLong_FromUnsignedLongLong(total_bases);
            if (summed_error_obj == NULL || count_obj == NULL) {
                Py_DECREF(result);
                return PyErr_NoMemory();
            }
            PyList_SetItem(summed_error_list, j, summed_error_obj);
            PyList_SetItem(count_list, j, count_obj);
        }
        PyTuple_SetItem(entry, 0, tile_id);
        PyTuple_SetItem(entry, 1, summed_error_list);
        PyTuple_SetItem(entry, 2, count_list);
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
    {"add_record_array", (PyCFunction)PerTileQuality_add_record_array,
     PerTileQuality_add_record_array_method,
     PerTileQuality_add_record_array__doc__},
    {"get_tile_counts", (PyCFunction)PerTileQuality_get_tile_counts,
     PerTileQuality_get_tile_counts_method, PerTileQuality_get_tile_counts__doc__},
    {NULL},
};

static PyMemberDef PerTileQuality_members[] = {
    {"max_length", T_PYSSIZET, offsetof(PerTileQuality, max_length), READONLY,
     "The length of the longest read"},
    {"number_of_reads", T_ULONGLONG, offsetof(PerTileQuality, number_of_reads),
     READONLY, "The total amount of reads counted"},
    {"skipped_reason", T_OBJECT, offsetof(PerTileQuality, skipped_reason), READONLY,
     "What the reason is for skipping the module if skipped."
     "Set to None if not skipped."},
    {NULL},
};

static PyType_Slot PerTileQuality_slots[] = {
    {Py_tp_dealloc, (destructor)PerTileQuality_dealloc},
    {Py_tp_new, (newfunc)PerTileQuality__new__},
    {Py_tp_members, PerTileQuality_members},
    {Py_tp_methods, PerTileQuality_methods},
    {0, NULL},
};

static PyType_Spec PerTileQuality_spec = {
    .name = "_qc.PerTileQuality",
    .basicsize = sizeof(PerTileQuality),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT,
    .slots = PerTileQuality_slots,
};

/**********************
 * TWOBIT CONVERSIONS *
 **********************/

/* Most functions moved to function_dispatch.h */

static void
kmer_to_sequence(uint64_t kmer, size_t k, uint8_t *sequence)
{
    static uint8_t nucs[4] = {'A', 'C', 'G', 'T'};
    for (size_t i = k; i > 0; i -= 1) {
        size_t nuc = kmer & 3;  // 3 == 0b11
        sequence[i - 1] = nucs[nuc];
        kmer >>= 2;
    }
}

/****************************
 * OverrepresentedSequences *
 ****************************/

/* A module that cuts the sequence in bits of k size. The canonical (lowest)
   representation of the bit is used.

   k should be an uneven number (so there is always a canonical kmer) and k
   should be 31 or lower so it can fit into a 64-bit integer. Then Thomas
   Wang's integer hash can be used to store the sequence in a hash table
   having the hash function both as a hash and as the storage for the sequence.
*/

#define DEFAULT_MAX_UNIQUE_FRAGMENTS 5000000
#define DEFAULT_FRAGMENT_LENGTH 21
#define DEFAULT_UNIQUE_SAMPLE_EVERY 8
#define DEFAULT_BASES_FROM_START 100
#define DEFAULT_BASES_FROM_END 100

typedef struct _OverrepresentedSequencesStruct {
    PyObject_HEAD
    size_t fragment_length;
    uint64_t number_of_sequences;
    uint64_t sampled_sequences;
    uint64_t staging_hash_table_size;
    uint64_t *staging_hash_table;
    uint64_t hash_table_size;
    uint64_t *hashes;
    uint32_t *counts;
    uint64_t max_unique_fragments;
    uint64_t number_of_unique_fragments;
    uint64_t total_fragments;
    size_t sample_every;
    Py_ssize_t fragments_from_start;
    Py_ssize_t fragments_from_end;
} OverrepresentedSequences;

static void
OverrepresentedSequences_dealloc(OverrepresentedSequences *self)
{
    PyMem_Free(self->staging_hash_table);
    PyMem_Free(self->hashes);
    PyMem_Free(self->counts);
    PyTypeObject *tp = Py_TYPE((PyObject *)self);
    PyObject_Free(self);
    Py_XDECREF((PyObject *)tp);
}

static PyObject *
OverrepresentedSequences__new__(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    Py_ssize_t max_unique_fragments = DEFAULT_MAX_UNIQUE_FRAGMENTS;
    Py_ssize_t fragment_length = DEFAULT_FRAGMENT_LENGTH;
    Py_ssize_t sample_every = DEFAULT_UNIQUE_SAMPLE_EVERY;
    Py_ssize_t bases_from_start = DEFAULT_BASES_FROM_START;
    Py_ssize_t bases_from_end = DEFAULT_BASES_FROM_END;
    static char *kwargnames[] = {"max_unique_fragments", "fragment_length",
                                 "sample_every",         "bases_from_start",
                                 "bases_from_end",       NULL};
    static char *format = "|nnnnn:OverrepresentedSequences";
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, format, kwargnames, &max_unique_fragments,
            &fragment_length, &sample_every, &bases_from_start, &bases_from_end)) {
        return NULL;
    }
    if (max_unique_fragments < 1) {
        PyErr_Format(PyExc_ValueError,
                     "max_unique_fragments should be at least 1, got: %zd",
                     max_unique_fragments);
        return NULL;
    }
    if ((fragment_length & 1) == 0 || fragment_length > 31 || fragment_length < 3) {
        PyErr_Format(PyExc_ValueError,
                     "fragment_length must be between 3 and 31 and be an "
                     "uneven number, got: %zd",
                     fragment_length);
        return NULL;
    }
    if (sample_every < 1) {
        PyErr_Format(PyExc_ValueError,
                     "sample_every must be 1 or greater. Got %zd", sample_every);
        return NULL;
    }
    if (bases_from_start < 0) {
        bases_from_start = UINT32_MAX;
    }
    if (bases_from_end < 0) {
        bases_from_end = UINT32_MAX;
    }
    /* If size is a power of 2, the modulo HASH_TABLE_SIZE can be optimised to
       a bitwise AND. Using 1.5 times as a base we ensure that the hashtable is
       utilized for at most 2/3. (Increased business degrades performance.) */
    uint64_t hash_table_bits = (uint64_t)(log2(max_unique_fragments * 1.5) + 1);
    uint64_t hash_table_size = 1 << hash_table_bits;
    uint64_t *hashes = PyMem_Calloc(hash_table_size, sizeof(uint64_t));
    uint32_t *counts = PyMem_Calloc(hash_table_size, sizeof(uint32_t));
    if ((hashes == NULL) || (counts == NULL)) {
        PyMem_Free(hashes);
        PyMem_Free(counts);
        return PyErr_NoMemory();
    }
    OverrepresentedSequences *self = PyObject_New(OverrepresentedSequences, type);
    if (self == NULL) {
        PyMem_Free(hashes);
        PyMem_Free(counts);
        return PyErr_NoMemory();
    }
    self->number_of_sequences = 0;
    self->sampled_sequences = 0;
    self->number_of_unique_fragments = 0;
    self->max_unique_fragments = max_unique_fragments;
    self->hash_table_size = hash_table_size;
    self->total_fragments = 0;
    self->fragment_length = fragment_length;
    self->staging_hash_table_size = 0;
    self->staging_hash_table = NULL;
    self->hashes = hashes;
    self->counts = counts;
    self->sample_every = sample_every;
    self->fragments_from_start =
        (bases_from_start + fragment_length - 1) / fragment_length;
    self->fragments_from_end =
        (bases_from_end + fragment_length - 1) / fragment_length;
    return (PyObject *)self;
}

static void
Sequence_duplication_insert_hash(OverrepresentedSequences *self, uint64_t hash)
{
    uint64_t hash_to_index_int = self->hash_table_size - 1;
    uint64_t *hashes = self->hashes;
    uint32_t *counts = self->counts;
    size_t index = hash & hash_to_index_int;

    while (1) {
        uint64_t hash_entry = hashes[index];
        if (hash_entry == 0) {
            if (self->number_of_unique_fragments < self->max_unique_fragments) {
                hashes[index] = hash;
                counts[index] = 1;
                self->number_of_unique_fragments += 1;
            }
            break;
        }
        else if (hash_entry == hash) {
            counts[index] += 1;
            break;
        }
        index += 1;
        /* Make sure the index round trips when it reaches hash_table_size.*/
        index &= hash_to_index_int;
    }
}

static int
OverrepresentedSequences_resize_staging(OverrepresentedSequences *self,
                                        uint64_t new_size)
{
    if (new_size <= self->staging_hash_table_size) {
        return 0;
    }
    uint64_t *tmp =
        PyMem_Realloc(self->staging_hash_table, new_size * sizeof(uint64_t));
    if (tmp == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    self->staging_hash_table = tmp;
    self->staging_hash_table_size = new_size;
    return 0;
}

static inline void
add_to_staging(uint64_t *staging_hash_table, uint64_t staging_hash_table_size,
               uint64_t hash)
{
    /* Works because size is a power of 2 */
    uint64_t hash_to_index_int = staging_hash_table_size - 1;
    uint64_t index = hash & hash_to_index_int;
    while (true) {
        uint64_t current_entry = staging_hash_table[index];
        if (current_entry == 0) {
            staging_hash_table[index] = hash;
            break;
        }
        else if (current_entry == hash) {
            break;
        }
        index += 1;
        index &= hash_to_index_int;
    }
    return;
}

/* To be used in the sequence duplication part */
// clang-format off
static const uint8_t NUCLEOTIDE_TO_TWOBIT[128] = {
// Control characters
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
// Interpunction numbers etc
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
//     A, B, C, D, E, F, G, H, I, J, K, L, M, N, O,
    4, 0, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 8, 4,
//  P, Q, R, S, T, U, V, W, X, Y, Z,
    4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
//     a, b, c, d, e, f, g, h, i, j, k, l, m, n, o,
    4, 0, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 8, 4,
//  p, q, r, s, t, u, v, w, x, y, z,
    4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
};
// clang-format on

#define TWOBIT_UNKNOWN_CHAR -1
#define TWOBIT_N_CHAR -2
#define TWOBIT_SUCCESS 0

static uint64_t
reverse_complement_kmer(uint64_t kmer, uint64_t k)
{
    // Invert all the bits, with 0,1,2,3 == A,C,G,T this automatically is the
    // complement.
    uint64_t comp = ~kmer;
    // Progressively swap all the twobits inplace.
    uint64_t revcomp = (comp << 32) | (comp >> 32);
    revcomp = ((revcomp & 0xFFFF0000FFFF0000ULL) >> 16) |
              ((revcomp & 0x0000FFFF0000FFFFULL) << 16);
    revcomp = ((revcomp & 0xFF00FF00FF00FF00ULL) >> 8) |
              ((revcomp & 0x00FF00FF00FF00FFULL) << 8);
    /* Compiler properly recognizes the above as a byteswap and will simplify
       using the bswap instruction. */
    revcomp = ((revcomp & 0xF0F0F0F0F0F0F0F0ULL) >> 4) |
              ((revcomp & 0x0F0F0F0F0F0F0F0FULL) << 4);
    revcomp = ((revcomp & 0xCCCCCCCCCCCCCCCCULL) >> 2) |
              ((revcomp & 0x3333333333333333ULL) << 2);
    // If k < 32, the empty twobit slots will have ended up at the least
    // significant bits. Use a shift to move them to the highest bits again.
    return revcomp >> (64 - (k * 2));
}

static int64_t
sequence_to_canonical_kmer_default(uint8_t *sequence, uint64_t k)
{
    uint64_t kmer = 0;
    size_t all_nucs = 0;
    int64_t i = 0;
    int64_t vector_end = k - 4;
    for (i = 0; i < vector_end; i += 4) {
        size_t nuc0 = NUCLEOTIDE_TO_TWOBIT[sequence[i]];
        size_t nuc1 = NUCLEOTIDE_TO_TWOBIT[sequence[i + 1]];
        size_t nuc2 = NUCLEOTIDE_TO_TWOBIT[sequence[i + 2]];
        size_t nuc3 = NUCLEOTIDE_TO_TWOBIT[sequence[i + 3]];
        all_nucs |= (nuc0 | nuc1 | nuc2 | nuc3);
        uint64_t kchunk = ((nuc0 << 6) | (nuc1 << 4) | (nuc2 << 2) | (nuc3));
        kmer <<= 8;
        kmer |= kchunk;
    }
    for (; i < (int64_t)k; i++) {
        size_t nuc = NUCLEOTIDE_TO_TWOBIT[sequence[i]];
        all_nucs |= nuc;
        kmer <<= 2;
        kmer |= nuc;
    }
    if (all_nucs > 3) {
        if (all_nucs & 4) {
            return TWOBIT_UNKNOWN_CHAR;
        }
        if (all_nucs & 8) {
            return TWOBIT_N_CHAR;
        }
    }
    uint64_t revcomp_kmer = reverse_complement_kmer(kmer, k);
    // If k is uneven there can be no ambiguity
    if (revcomp_kmer > kmer) {
        return kmer;
    }
    return revcomp_kmer;
}

static int64_t (*sequence_to_canonical_kmer)(uint8_t *sequence, uint64_t k) =
    sequence_to_canonical_kmer_default;

#if COMPILER_HAS_TARGETED_DISPATCH && BUILD_IS_X86_64
__attribute__((__target__("avx2"))) static int64_t
sequence_to_canonical_kmer_avx2(uint8_t *sequence, uint64_t k)
{
    /* By using a load mask, at most 3 extra bytes are loaded. Given that a
       sequence in sequali always ends with \n+\n this should not trigger
       invalid memory access.*/
    __m256i load_mask = _mm256_cmpgt_epi32(
        _mm256_add_epi32(_mm256_set1_epi32((k + 3) / 4),
                         _mm256_setr_epi32(0, -1, -2, -3, -4, -5, -6, -7)),
        _mm256_setzero_si256());
    __m256i seq_vec_raw = _mm256_maskload_epi32((int *)sequence, load_mask);
    /* Use only the last 3 bits to create indices from 0-15. A,C,G and T are
        distinct in the last 3 bits. This will yield results for any
        input. The non-ACGT check is performed at the end of the function.
    */
    __m256i indices_vec = _mm256_and_si256(_mm256_set1_epi8(7), seq_vec_raw);
    /* Use the shufb instruction to convert the 0-7 indices to corresponding
       ACGT twobit representation. Everything non C, G, T will be 0, the same
       as A. */
    __m256i twobit_vec =
        _mm256_shuffle_epi8(_mm256_setr_epi8(
                                /*     A,  , C, T,  ,  , G */
                                0, 0, 0, 1, 3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 1, 3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0),
                            indices_vec);
    /* Now the twobits need to be shifted to be in the position of their byte.
     * For each group of 4, the first entry must be shifted by 6, the second by
     * 4, the third by 2 and the last not shifted.
     * To do this, an alternating mask ff00 is used to select only one byte of
     * each byte pair. This byte is shifted by 2. This leads to a vector were
     * bytes are shifted by 2, 0, 2, 0 etc.
     * After that the first two bytes of each four byte group are selected and
     * shifted by 4. This results in a vector were all bytes are shifted
     * 6, 4, 2, 0.
     */
    __m256i alternate_byte_select = _mm256_set1_epi16(0x00ff);
    __m256i alternate_word_select = _mm256_set1_epi32(0x0000ffff);
    __m256i first_shift = _mm256_blendv_epi8(
        twobit_vec, _mm256_slli_epi16(twobit_vec, 2), alternate_byte_select);
    __m256i twobit_shifted_vec = _mm256_blendv_epi8(
        first_shift, _mm256_slli_epi16(first_shift, 4), alternate_word_select);
    /* Now the groups of four need to be bitwise ORred together. Due to the
       way the data is prepared ADD and OR have the same effect. We can use
       _mm256_sad_epu8 instruction with a zero function to horizontally add
       8-bit integers. Since this adds 8-bit integers in groups of 8, we use
       a mask to select only 4 bytes. We can then use a shift and a or to
       get all resulting integers into one vector again.
    */
    __m256i four_select_mask = _mm256_set1_epi64x(0x00000000FFFFFFFF);
    __m256i upper_twobit =
        _mm256_sad_epu8(_mm256_and_si256(four_select_mask, twobit_shifted_vec),
                        _mm256_setzero_si256());
    __m256i lower_twobit = _mm256_sad_epu8(
        _mm256_andnot_si256(four_select_mask, twobit_shifted_vec),
        _mm256_setzero_si256());
    __m256i combined_twobit =
        _mm256_or_si256(_mm256_bslli_epi128(upper_twobit, 1), lower_twobit);

    /* The following instructions arrange the 8 resulting twobit bytes in the
       correct order to be extracted as a 64 bit integer. */
    __m256i packed_twobit = _mm256_shuffle_epi8(
        combined_twobit,
        _mm256_setr_epi8(8, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, 8, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, -1, -1));
    __m256i shuffled_twobit = _mm256_permutevar8x32_epi32(
        packed_twobit, _mm256_setr_epi32(4, 0, 7, 7, 7, 7, 7, 7));
    uint64_t kmer = _mm_cvtsi128_si64(_mm256_castsi256_si128(shuffled_twobit));
    kmer = kmer >> (64 - (k * 2));

    /* NON-ACGT CHECK*/

    /* In order to mask only k characters.
      Create an array with only k. Create an array with 0, -1, -2, -3 etc.
      Add k array to descending array. If k=2 descending array will be,
      2, 1, 0, -1, etc. cmpgt with zero array results in, yes, yes, no, no etc.
      First two characters masked with k=2.
   */
    __m256i seq_mask = _mm256_cmpgt_epi8(
        _mm256_add_epi8(
            _mm256_set1_epi8(k),
            _mm256_setr_epi8(0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11,
                             -12, -13, -14, -15, -16, -17, -18, -19, -20, -21,
                             -22, -23, -24, -25, -26, -27, -28, -29, -30, -31)),
        _mm256_setzero_si256());
    /* Mask all characters not of interest as A to not false positively trigger
       the non-ACGT detection */
    __m256i seq_vec =
        _mm256_blendv_epi8(_mm256_set1_epi8('A'), seq_vec_raw, seq_mask);
    /* 32 is the ASCII lowercase bit. Use and not to make everything upper case. */
    __m256i seq_vec_upper = _mm256_andnot_si256(_mm256_set1_epi8(32), seq_vec);
    __m256i ACGT_vec = _mm256_or_si256(
        _mm256_or_si256(_mm256_cmpeq_epi8(seq_vec_upper, _mm256_set1_epi8('A')),
                        _mm256_cmpeq_epi8(seq_vec_upper, _mm256_set1_epi8('C'))),
        _mm256_or_si256(_mm256_cmpeq_epi8(seq_vec_upper, _mm256_set1_epi8('G')),
                        _mm256_cmpeq_epi8(seq_vec_upper, _mm256_set1_epi8('T'))

                            ));
    /* Bitwise not of ACGT_vec should be all 0. */
    int all_characters_acgt = _mm256_testc_si256(ACGT_vec, _mm256_set1_epi8(-1));
    if (all_characters_acgt) {
        uint64_t revcomp_kmer = reverse_complement_kmer(kmer, k);
        if (revcomp_kmer < kmer) {
            return revcomp_kmer;
        }
        return kmer;
    }
    __m256i N_vec = _mm256_cmpeq_epi8(seq_vec_upper, _mm256_set1_epi8('N'));
    int all_characters_acgtn = _mm256_testc_si256(
        _mm256_or_si256(ACGT_vec, N_vec), _mm256_set1_epi8(-1));
    if (all_characters_acgtn) {
        return TWOBIT_N_CHAR;
    }
    return TWOBIT_UNKNOWN_CHAR;
}

/* Constructor runs at dynamic load time */
__attribute__((constructor)) static void
sequence_to_canonical_kmer_init_func_ptr(void)
{
    if (__builtin_cpu_supports("avx2")) {
        sequence_to_canonical_kmer = sequence_to_canonical_kmer_avx2;
    }
    else {
        sequence_to_canonical_kmer = sequence_to_canonical_kmer_default;
    }
}
#endif

static int
OverrepresentedSequences_add_meta(OverrepresentedSequences *self,
                                  struct FastqMeta *meta)
{
    if (self->number_of_sequences % self->sample_every != 0) {
        self->number_of_sequences += 1;
        return 0;
    }
    self->sampled_sequences += 1;
    self->number_of_sequences += 1;
    Py_ssize_t sequence_length = meta->sequence_length;
    Py_ssize_t fragment_length = self->fragment_length;
    size_t fragments = 0;
    if (sequence_length < fragment_length) {
        return 0;
    }
    uint8_t *sequence = meta->record_start + meta->sequence_offset;
    /* A full fragment at the beginning and the end is desired so that adapter
       fragments at the beginning and end do not get added to the hash table in
       a lot of different frames. To do so sample from the beginning and end
       with a little overlap in the middle

                                     | <- mid_point
       sequence    ==========================================
       from front  |------||------||------|
       from back                     |------||------||------|

       The mid_point is not the exact middle, but the middlish point were the
       back sequences start sampling.

       If the sequence length is exactly divisible by the fragment length, this
       results in exactly no overlap between front and back fragments, while
       still all of the sequence is being sampled.

       If the sequence is very large the amount of samples is taken is limited
       by a user-settable maximum.
    */
    /* Vader: Luke, Obi-Wan never told you about the algorithm...
       Luke:  He told me enough! It uses integer division!
       Vader: Yes Luke, we have to use integer division.
       Luke:  No, that's not true! The compiler can optimize integer division
              away for constants!
       Vader: Search your feelings Luke! The fragment length has to be user
              settable!
       Luke:  NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
    */
    Py_ssize_t max_fragments =
        (sequence_length + fragment_length - 1) / fragment_length;
    Py_ssize_t from_mid_point_fragments = max_fragments / 2;
    Py_ssize_t max_start_fragments = max_fragments - from_mid_point_fragments;
    Py_ssize_t fragments_from_start =
        Py_MIN(self->fragments_from_start, max_start_fragments);
    Py_ssize_t fragments_from_end =
        Py_MIN(self->fragments_from_end, from_mid_point_fragments);
    Py_ssize_t total_fragments = fragments_from_start + fragments_from_end;
    size_t staging_hash_bits = (size_t)ceil(log2((double)total_fragments * 1.5));
    size_t staging_hash_size = 1ULL << staging_hash_bits;
    if (staging_hash_size > self->staging_hash_table_size) {
        if (OverrepresentedSequences_resize_staging(self, staging_hash_size) < 0) {
            return -1;
        }
    }
    uint64_t *staging_hash_table = self->staging_hash_table;
    memset(staging_hash_table, 0, staging_hash_size * sizeof(uint64_t));

    Py_ssize_t start_end = fragments_from_start * fragment_length;
    Py_ssize_t end_start =
        sequence_length - (fragments_from_end * fragment_length);
    bool warn_unknown = false;
    // Sample front sequences
    for (Py_ssize_t i = 0; i < start_end; i += fragment_length) {
        int64_t kmer = sequence_to_canonical_kmer(sequence + i, fragment_length);
        if (kmer < 0) {
            if (kmer == TWOBIT_UNKNOWN_CHAR) {
                warn_unknown = true;
            }
            continue;
        }
        fragments += 1;
        uint64_t hash = wanghash64(kmer);
        add_to_staging(staging_hash_table, staging_hash_size, hash);
    }

    // Sample back sequences
    for (Py_ssize_t i = end_start; i < sequence_length; i += fragment_length) {
        int64_t kmer = sequence_to_canonical_kmer(sequence + i, fragment_length);
        if (kmer < 0) {
            if (kmer == TWOBIT_UNKNOWN_CHAR) {
                warn_unknown = true;
            }
            continue;
        }
        fragments += 1;
        uint64_t hash = wanghash64(kmer);
        add_to_staging(staging_hash_table, staging_hash_size, hash);
    }
    for (size_t i = 0; i < staging_hash_size; i++) {
        uint64_t hash = staging_hash_table[i];
        if (hash != 0) {
            Sequence_duplication_insert_hash(self, hash);
        }
    }
    if (warn_unknown) {
        PyObject *culprit =
            PyUnicode_DecodeASCII((char *)sequence, sequence_length, NULL);
        PyErr_WarnFormat(
            PyExc_UserWarning, 1,
            "Sequence contains a chacter that is not A, C, G, T or N: %R",
            culprit);
        Py_DECREF(culprit);
    }
    self->total_fragments += fragments;
    return 0;
}

PyDoc_STRVAR(OverrepresentedSequences_add_read__doc__,
             "add_read($self, read, /)\n"
             "--\n"
             "\n"
             "Add a read to the duplication module. \n"
             "\n"
             "  read\n"
             "    A FastqRecordView object.\n");

#define OverrepresentedSequences_add_read_method METH_O

static PyObject *
OverrepresentedSequences_add_read(OverrepresentedSequences *self,
                                  FastqRecordView *read)
{
    int is_view = is_FastqRecordView(self, read);
    if (is_view == -1) {
        return NULL;
    }
    else if (is_view == 0) {
        PyErr_Format(PyExc_TypeError,
                     "read should be a FastqRecordView object, got %R",
                     Py_TYPE((PyObject *)read));
        return NULL;
    }
    if (OverrepresentedSequences_add_meta(self, &read->meta) != 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(OverrepresentedSequences_add_record_array__doc__,
             "add_record_array($self, record_array, /)\n"
             "--\n"
             "\n"
             "Add a record_array to the duplication module. \n"
             "\n"
             "  record_array\n"
             "    A FastqRecordArrayView object.\n");

#define OverrepresentedSequences_add_record_array_method METH_O

static PyObject *
OverrepresentedSequences_add_record_array(OverrepresentedSequences *self,
                                          FastqRecordArrayView *record_array)
{
    int is_record_array = is_FastqRecordArrayView(self, record_array);
    if (is_record_array == -1) {
        return NULL;
    }
    else if (is_record_array == 0) {
        PyErr_Format(
            PyExc_TypeError,
            "record_array should be a FastqRecordArrayView object, got %R",
            Py_TYPE((PyObject *)record_array));
        return NULL;
    }
    Py_ssize_t number_of_records = Py_SIZE((PyObject *)record_array);
    struct FastqMeta *records = record_array->records;
    for (Py_ssize_t i = 0; i < number_of_records; i++) {
        if (OverrepresentedSequences_add_meta(self, records + i) != 0) {
            return NULL;
        }
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(OverrepresentedSequences_sequence_counts__doc__,
             "sequence_counts($self, /)\n"
             "--\n"
             "\n"
             "Get a dictionary with sequence counts \n");

#define OverrepresentedSequences_sequence_counts_method METH_NOARGS

static PyObject *
OverrepresentedSequences_sequence_counts(OverrepresentedSequences *self,
                                         PyObject *Py_UNUSED(ignore))
{
    PyObject *count_dict = PyDict_New();
    if (count_dict == NULL) {
        return PyErr_NoMemory();
    }
    uint64_t *hashes = self->hashes;
    uint32_t *counts = self->counts;
    uint64_t hash_table_size = self->hash_table_size;
    Py_ssize_t fragment_length = self->fragment_length;
    uint8_t seq_store[32];
    memset(seq_store, 0, sizeof(seq_store));
    for (size_t i = 0; i < hash_table_size; i += 1) {
        uint64_t entry_hash = hashes[i];
        if (entry_hash == 0) {
            continue;
        }
        PyObject *count_obj = PyLong_FromUnsignedLong(counts[i]);
        if (count_obj == NULL) {
            goto error;
        }
        uint64_t kmer = wanghash64_inverse(entry_hash);
        kmer_to_sequence(kmer, fragment_length, seq_store);
        PyObject *key =
            PyUnicode_DecodeASCII((char *)seq_store, fragment_length, NULL);
        if (key == NULL) {
            Py_DECREF(count_obj);
            goto error;
        }
        memset(seq_store, 0, sizeof(seq_store));
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

PyDoc_STRVAR(
    OverrepresentedSequences_overrepresented_sequences__doc__,
    "overrepresented_sequences($self, threshold=0.001)\n"
    "--\n"
    "\n"
    "Return a list of tuples with the count, fraction and the sequence. The "
    "list is "
    "sorted in reverse order with the most common sequence on top.\n"
    "\n"
    "  threshold_fraction\n"
    "    The fraction at which a sequence is considered overrepresented.\n"
    "  min_threshold\n"
    "    the minimum threshold to uphold. Overrides the minimum number based "
    "on "
    "    the threshold_fraction if it is higher. Useful for files with very "
    "low "
    "    numbers of sequences."
    "  max_threshold\n"
    "    the maximum threshold to uphold. Overrides the minimum number based "
    "on "
    "    the threshold_fraction if it is lower. Useful for files with very "
    "high "
    "    numbers of sequences.");

#define OverrepresentedSequences_overrepresented_sequences_method \
    METH_VARARGS | METH_KEYWORDS

static PyObject *
OverrepresentedSequences_overrepresented_sequences(OverrepresentedSequences *self,
                                                   PyObject *args,
                                                   PyObject *kwargs)
{
    double threshold = 0.0001;  // 0.01 %
    Py_ssize_t min_threshold = 1;
    Py_ssize_t max_threshold = PY_SSIZE_T_MAX;
    static char *kwargnames[] = {"threshold_fraction", "min_threshold",
                                 "max_threshold", NULL};
    static char *format =
        "|dnn:OverrepresentedSequences.overrepresented_sequences";
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, kwargnames, &threshold,
                                     &min_threshold, &max_threshold)) {
        return NULL;
    }
    if ((threshold < 0.0) || (threshold > 1.0)) {
        // PyErr_Format has no direct way to represent floats
        PyObject *threshold_obj = PyFloat_FromDouble(threshold);
        PyErr_Format(PyExc_ValueError,
                     "threshold_fraction must be between 0.0 and 1.0 got, %R",
                     threshold_obj, threshold);
        Py_XDECREF(threshold_obj);
        return NULL;
    }
    if (min_threshold < 1) {
        PyErr_Format(PyExc_ValueError,
                     "min_threshold must be at least 1, got %zd", min_threshold);
        return NULL;
    }
    if (max_threshold < 1) {
        PyErr_Format(PyExc_ValueError,
                     "max_threshold must be at least 1, got %zd", max_threshold);
        return NULL;
    }
    if (max_threshold < min_threshold) {
        PyErr_Format(
            PyExc_ValueError,
            "max_threshold (%zd) must be greater than min_threshold (%zd)",
            max_threshold, min_threshold);
    }

    PyObject *result = PyList_New(0);
    if (result == NULL) {
        return NULL;
    }

    uint64_t sampled_sequences = self->sampled_sequences;
    Py_ssize_t hit_theshold = ceil(threshold * sampled_sequences);
    hit_theshold = Py_MAX(min_threshold, hit_theshold);
    hit_theshold = Py_MIN(max_threshold, hit_theshold);
    uint64_t minimum_hits = hit_theshold;
    uint64_t *hashes = self->hashes;
    uint32_t *counts = self->counts;
    size_t hash_table_size = self->hash_table_size;
    Py_ssize_t fragment_length = self->fragment_length;
    uint8_t seq_store[32];
    memset(seq_store, 0, sizeof(seq_store));
    for (size_t i = 0; i < hash_table_size; i += 1) {
        uint32_t count = counts[i];
        if (count >= minimum_hits) {
            uint64_t entry_hash = hashes[i];
            uint64_t kmer = wanghash64_inverse(entry_hash);
            kmer_to_sequence(kmer, fragment_length, seq_store);
            PyObject *entry_tuple = Py_BuildValue(
                "(KdU#)", count,
                (double)((double)count / (double)sampled_sequences), seq_store,
                (Py_ssize_t)fragment_length);
            memset(seq_store, 0, sizeof(seq_store));
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

static PyMethodDef OverrepresentedSequences_methods[] = {
    {"add_read", (PyCFunction)OverrepresentedSequences_add_read,
     OverrepresentedSequences_add_read_method,
     OverrepresentedSequences_add_read__doc__},
    {"add_record_array", (PyCFunction)OverrepresentedSequences_add_record_array,
     OverrepresentedSequences_add_record_array_method,
     OverrepresentedSequences_add_record_array__doc__},
    {"sequence_counts", (PyCFunction)OverrepresentedSequences_sequence_counts,
     OverrepresentedSequences_sequence_counts_method,
     OverrepresentedSequences_sequence_counts__doc__},
    {"overrepresented_sequences",
     (PyCFunction)(void (*)(void))OverrepresentedSequences_overrepresented_sequences,
     OverrepresentedSequences_overrepresented_sequences_method,
     OverrepresentedSequences_overrepresented_sequences__doc__},
    {NULL},
};

static PyMemberDef OverrepresentedSequences_members[] = {
    {"number_of_sequences", T_ULONGLONG,
     offsetof(OverrepresentedSequences, number_of_sequences), READONLY,
     "The total number of sequences submitted."},
    {"sampled_sequences", T_ULONGLONG,
     offsetof(OverrepresentedSequences, sampled_sequences), READONLY,
     "The total number of sequences that were analysed."},
    {"collected_unique_fragments", T_ULONGLONG,
     offsetof(OverrepresentedSequences, number_of_unique_fragments), READONLY,
     "The number of unique fragments collected."},
    {"max_unique_fragments", T_ULONGLONG,
     offsetof(OverrepresentedSequences, max_unique_fragments), READONLY,
     "The maximum number of unique sequences stored in the object."},
    {"fragment_length", T_BYTE, offsetof(OverrepresentedSequences, fragment_length),
     READONLY, "The length of the sampled sequences"},
    {"sample_every", T_PYSSIZET, offsetof(OverrepresentedSequences, sample_every),
     READONLY, "One in this many reads is sampled"},
    {"total_fragments", T_ULONGLONG,
     offsetof(OverrepresentedSequences, total_fragments), READONLY,
     "Total number of fragments."},
    {NULL},
};

static PyType_Slot OverrepresentedSequences_slots[] = {
    {Py_tp_dealloc, (destructor)OverrepresentedSequences_dealloc},
    {Py_tp_new, (newfunc)OverrepresentedSequences__new__},
    {Py_tp_members, OverrepresentedSequences_members},
    {Py_tp_methods, OverrepresentedSequences_methods},
    {0, NULL},
};

static PyType_Spec OverrepresentedSequences_spec = {
    .name = "_qc.OverrepresentedSequences",
    .basicsize = sizeof(OverrepresentedSequences),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT,
    .slots = OverrepresentedSequences_slots,
};

/*******************
 * DEDUP ESTIMATOR *
 *******************/
/*
Based on the following paper:
Estimating Duplication by Content-based Sampling
Fei Xie, Michael Condict, Sandip Shete
https://www.usenix.org/system/files/conference/atc13/atc13-xie.pdf
*/

/*
Store 1 million fingerprints. This requires 24MB which balloons to 48MB when
creating a new table. Between 500,000 and 1,000,000 sequences will lead to a
quite accurate result.
*/
#define DEFAULT_DEDUP_MAX_STORED_FINGERPRINTS 1000000

/*
Avoid the beginning and end of the sequence by at most 64 bp to avoid
any adapters. Take the 8 bp after the start offset and the 8 bp before
the end offset. This creates a small 16 bp fingerprint. Hash it using
MurmurHash. 16 bp is small and therefore relatively insensitive to
sequencing errors while still offering 4^16 or 4 billion distinct
fingerprints.
*/
#define DEFAULT_FINGERPRINT_FRONT_SEQUENCE_LENGTH 8
#define DEFAULT_FINGERPRINT_BACK_SEQUENCE_LENGTH 8
#define DEFAULT_FINGERPRINT_FRONT_SEQUENCE_OFFSET 64
#define DEFAULT_FINGERPRINT_BACK_SEQUENCE_OFFSET 64

// Use packing at the 4-byte boundary to save 4 bytes of storage.
#pragma pack(4)
struct EstimatorEntry {
    uint64_t hash;
    // 32 bits allows storing 4 billion counts. This should never overflow in practice.
    uint32_t count;
};

typedef struct _DedupEstimatorStruct {
    PyObject_HEAD
    size_t modulo_bits;
    size_t hash_table_size;
    size_t max_stored_entries;
    size_t stored_entries;
    size_t front_sequence_length;
    size_t front_sequence_offset;
    size_t back_sequence_length;
    size_t back_sequence_offset;
    uint8_t *fingerprint_store;
    struct EstimatorEntry *hash_table;
} DedupEstimator;

static void
DedupEstimator_dealloc(DedupEstimator *self)
{
    PyMem_Free(self->hash_table);
    PyMem_Free(self->fingerprint_store);
    PyTypeObject *tp = Py_TYPE((PyObject *)self);
    PyObject_Free(self);
    Py_XDECREF((PyObject *)tp);
}

static PyObject *
DedupEstimator__new__(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    Py_ssize_t max_stored_fingerprints = DEFAULT_DEDUP_MAX_STORED_FINGERPRINTS;
    Py_ssize_t front_sequence_length = DEFAULT_FINGERPRINT_FRONT_SEQUENCE_LENGTH;
    Py_ssize_t front_sequence_offset = DEFAULT_FINGERPRINT_FRONT_SEQUENCE_OFFSET;
    Py_ssize_t back_sequence_length = DEFAULT_FINGERPRINT_BACK_SEQUENCE_LENGTH;
    Py_ssize_t back_sequence_offset = DEFAULT_FINGERPRINT_BACK_SEQUENCE_OFFSET;
    static char *kwargnames[] = {
        "max_stored_fingerprints", "front_sequence_length",
        "back_sequence_length",    "front_sequence_offset",
        "back_sequence_offset",    NULL};
    static char *format = "|n$nnnn:DedupEstimator";
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, format, kwargnames, &max_stored_fingerprints,
            &front_sequence_length, &back_sequence_length,
            &front_sequence_offset, &back_sequence_offset)) {
        return NULL;
    }

    if (max_stored_fingerprints < 100) {
        PyErr_Format(PyExc_ValueError,
                     "max_stored_fingerprints must be at least 100, not %zd",
                     max_stored_fingerprints);
        return NULL;
    }
    size_t hash_table_size_bits =
        (size_t)(log2(max_stored_fingerprints * 1.5) + 1);

    Py_ssize_t lengths_and_offsets[4] = {
        front_sequence_length,
        back_sequence_length,
        front_sequence_offset,
        back_sequence_offset,
    };
    for (size_t i = 0; i < 4; i++) {
        if (lengths_and_offsets[i] < 0) {
            PyErr_Format(PyExc_ValueError, "%s must be at least 0, got %zd.",
                         kwargnames[i + 1], lengths_and_offsets[i]);
            return NULL;
        }
    }
    size_t fingerprint_size = front_sequence_length + back_sequence_length;
    if (fingerprint_size == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "The sum of front_sequence_length and "
                        "back_sequence_length must be at least 0");
        return NULL;
    }

    size_t hash_table_size = 1ULL << hash_table_size_bits;
    uint8_t *fingerprint_store = PyMem_Malloc(fingerprint_size);
    if (fingerprint_store == NULL) {
        return PyErr_NoMemory();
    }
    struct EstimatorEntry *hash_table =
        PyMem_Calloc(hash_table_size, sizeof(struct EstimatorEntry));
    if (hash_table == NULL) {
        PyMem_Free(fingerprint_store);
        return PyErr_NoMemory();
    }
    DedupEstimator *self = PyObject_New(DedupEstimator, type);
    if (self == NULL) {
        PyMem_Free(fingerprint_store);
        PyMem_Free(hash_table);
        return PyErr_NoMemory();
    }
    self->front_sequence_length = front_sequence_length;
    self->front_sequence_offset = front_sequence_offset;
    self->back_sequence_length = back_sequence_length;
    self->back_sequence_offset = back_sequence_offset;
    self->fingerprint_store = fingerprint_store;
    self->hash_table_size = hash_table_size;
    // Get about 70% occupancy max
    self->max_stored_entries = max_stored_fingerprints;
    self->hash_table = hash_table;
    self->modulo_bits = 0;
    self->stored_entries = 0;
    return (PyObject *)self;
}

static int
DedupEstimator_increment_modulo(DedupEstimator *self)
{
    size_t next_modulo_bits = self->modulo_bits + 1;
    size_t next_ignore_mask = (1ULL << next_modulo_bits) - 1;
    struct EstimatorEntry *hash_table = self->hash_table;
    size_t hash_table_size = self->hash_table_size;
    size_t index_mask = hash_table_size - 1;
    size_t new_stored_entries = 0;
    struct EstimatorEntry *new_hash_table =
        PyMem_Calloc(hash_table_size, sizeof(struct EstimatorEntry));
    if (new_hash_table == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    for (size_t i = 0; i < hash_table_size; i++) {
        struct EstimatorEntry entry = hash_table[i];
        uint64_t hash = entry.hash;
        if (entry.count == 0 || hash & next_ignore_mask) {
            continue;
        }
        size_t new_index = (hash >> next_modulo_bits) & index_mask;
        while (true) {
            struct EstimatorEntry *current_entry = new_hash_table + new_index;
            if (current_entry->count == 0) {
                current_entry->hash = hash;
                current_entry->count = entry.count;
                break;
            }
            new_index += 1;
            new_index &= index_mask;
        }
        new_stored_entries += 1;
    }
    struct EstimatorEntry *tmp = self->hash_table;
    self->hash_table = new_hash_table;
    self->modulo_bits = next_modulo_bits;
    self->stored_entries = new_stored_entries;
    PyMem_Free(tmp);
    return 0;
}

static int
DedupEstimator_add_fingerprint(DedupEstimator *self, const uint8_t *fingerprint,
                               size_t fingerprint_length, uint64_t seed)
{
    uint64_t hash = MurmurHash3_x64_64(fingerprint, fingerprint_length, seed);
    size_t modulo_bits = self->modulo_bits;
    size_t ignore_mask = (1ULL << modulo_bits) - 1;
    if (hash & ignore_mask) {
        return 0;
    }
    size_t hash_table_size = self->hash_table_size;
    if (self->stored_entries >= self->max_stored_entries) {
        if (DedupEstimator_increment_modulo(self) != 0) {
            return -1;
        }
    }
    size_t index_mask = hash_table_size - 1;
    size_t index = (hash >> modulo_bits) & index_mask;
    struct EstimatorEntry *hash_table = self->hash_table;
    while (true) {
        struct EstimatorEntry *current_entry = hash_table + index;
        if (current_entry->count == 0) {
            current_entry->hash = hash;
            current_entry->count = 1;
            self->stored_entries += 1;
            break;
        }
        else if (current_entry->hash == hash) {
            current_entry->count += 1;
            break;
        }
        index += 1;
        index &= index_mask;
    }
    return 0;
}

static int
DedupEstimator_add_sequence_ptr(DedupEstimator *self, const uint8_t *sequence,
                                size_t sequence_length)
{
    size_t front_sequence_length = self->front_sequence_length;
    size_t back_sequence_length = self->back_sequence_length;
    size_t front_sequence_offset = self->front_sequence_offset;
    size_t back_sequence_offset = self->back_sequence_offset;
    size_t fingerprint_length = front_sequence_length + back_sequence_length;
    uint8_t *fingerprint = self->fingerprint_store;
    if (sequence_length <= fingerprint_length) {
        return DedupEstimator_add_fingerprint(self, sequence, sequence_length, 0);
    }
    uint64_t seed = sequence_length >> 6;
    size_t remainder = sequence_length - fingerprint_length;
    size_t front_offset = Py_MIN(remainder / 2, front_sequence_offset);
    size_t back_offset = Py_MIN(remainder / 2, back_sequence_offset);
    memcpy(fingerprint, sequence + front_offset, front_sequence_length);
    memcpy(fingerprint + front_sequence_length,
           sequence + sequence_length - (back_offset + back_sequence_length),
           back_sequence_length);
    return DedupEstimator_add_fingerprint(self, fingerprint,
                                          fingerprint_length, seed);
}

static int
DedupEstimator_add_sequence_pair_ptr(DedupEstimator *self,
                                     const uint8_t *sequence1,
                                     Py_ssize_t sequence_length1,
                                     const uint8_t *sequence2,
                                     Py_ssize_t sequence_length2)
{
    Py_ssize_t front_sequence_length = self->front_sequence_length;
    Py_ssize_t back_sequence_length = self->back_sequence_length;
    Py_ssize_t front_sequence_offset = self->front_sequence_offset;
    Py_ssize_t back_sequence_offset = self->back_sequence_offset;
    Py_ssize_t fingerprint_length = front_sequence_length + back_sequence_length;
    uint8_t *fingerprint = self->fingerprint_store;
    uint64_t seed = (sequence_length1 + sequence_length2) >> 6;

    // Ensure not more sequence is taken than available.
    front_sequence_length = Py_MIN(front_sequence_length, sequence_length1);
    // Ensure that the offset is not beyond the length of the sequence.
    Py_ssize_t front_offset = Py_MIN(
        front_sequence_offset, (sequence_length1 - front_sequence_length));
    // Same guarantees for sequence 2.
    back_sequence_length = Py_MIN(back_sequence_length, sequence_length2);
    Py_ssize_t back_offset =
        Py_MIN(back_sequence_offset, (sequence_length2 - back_sequence_length));

    memcpy(fingerprint, sequence1 + front_offset, front_sequence_length);
    memcpy(fingerprint + front_sequence_length, sequence2 + back_offset,
           back_sequence_length);
    return DedupEstimator_add_fingerprint(self, fingerprint,
                                          fingerprint_length, seed);
}

PyDoc_STRVAR(DedupEstimator_add_record_array__doc__,
             "add_record_array($self, record_array, /)\n"
             "--\n"
             "\n"
             "Add a record_array to the deduplication estimator. \n"
             "\n"
             "  record_array\n"
             "    A FastqRecordArrayView object.\n");

#define DedupEstimator_add_record_array_method METH_O

static PyObject *
DedupEstimator_add_record_array(DedupEstimator *self,
                                FastqRecordArrayView *record_array)
{
    int is_record_array = is_FastqRecordArrayView(self, record_array);
    if (is_record_array == -1) {
        return NULL;
    }
    else if (is_record_array == 0) {
        PyErr_Format(
            PyExc_TypeError,
            "record_array should be a FastqRecordArrayView object, got %R",
            Py_TYPE((PyObject *)record_array));
        return NULL;
    }
    Py_ssize_t number_of_records = Py_SIZE((PyObject *)record_array);
    struct FastqMeta *records = record_array->records;
    for (Py_ssize_t i = 0; i < number_of_records; i++) {
        struct FastqMeta *meta = records + i;
        uint8_t *sequence = meta->record_start + meta->sequence_offset;
        size_t sequence_length = meta->sequence_length;
        if (DedupEstimator_add_sequence_ptr(self, sequence, sequence_length) != 0) {
            return NULL;
        }
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(DedupEstimator_add_record_array_pair__doc__,
             "add_record_array_pair($self, record_array1, record_array2 /)\n"
             "--\n"
             "\n"
             "Add a pair of record arrays to the deduplication estimator. \n"
             "\n"
             "  record_array1\n"
             "    A FastqRecordArrayView object. First of read pair.\n"
             "  record_array2\n"
             "    A FastqRecordArrayView object. Second of read pair.\n");

#define DedupEstimator_add_record_array_pair_method METH_FASTCALL

static PyObject *
DedupEstimator_add_record_array_pair(DedupEstimator *self,
                                     PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2) {
        PyErr_Format(PyExc_TypeError,
                     "Dedupestimatorr.add_record_array_pair() "
                     "takes exactly two arguments (%zd given)",
                     nargs);
    }
    FastqRecordArrayView *record_array1 = (FastqRecordArrayView *)args[0];
    FastqRecordArrayView *record_array2 = (FastqRecordArrayView *)args[1];
    int is_record_array1 = is_FastqRecordArrayView(self, record_array1);
    if (is_record_array1 == -1) {
        return NULL;
    }
    else if (is_record_array1 == 0) {
        PyErr_Format(
            PyExc_TypeError,
            "record_array1 should be a FastqRecordArrayView object, got %R",
            Py_TYPE((PyObject *)record_array1));
        return NULL;
    }
    int is_record_array2 = is_FastqRecordArrayView(self, record_array2);
    if (is_record_array2 == -1) {
        return NULL;
    }
    else if (is_record_array2 == 0) {
        PyErr_Format(
            PyExc_TypeError,
            "record_array2 should be a FastqRecordArrayView object, got %R",
            Py_TYPE((PyObject *)record_array2));
        return NULL;
    }
    Py_ssize_t number_of_records = Py_SIZE((PyObject *)record_array1);
    if (Py_SIZE((PyObject *)record_array2) != number_of_records) {
        PyErr_Format(
            PyExc_ValueError,
            "record_array1 and record_array2 must be of the same size. "
            "Got %zd and %zd respectively.",
            number_of_records, Py_SIZE((PyObject *)record_array2));
    }
    struct FastqMeta *records1 = record_array1->records;
    struct FastqMeta *records2 = record_array2->records;
    for (Py_ssize_t i = 0; i < number_of_records; i++) {
        struct FastqMeta *meta1 = records1 + i;
        struct FastqMeta *meta2 = records2 + i;
        uint8_t *sequence1 = meta1->record_start + meta1->sequence_offset;
        uint8_t *sequence2 = meta2->record_start + meta2->sequence_offset;
        size_t sequence_length1 = meta1->sequence_length;
        size_t sequence_length2 = meta2->sequence_length;
        int ret = DedupEstimator_add_sequence_pair_ptr(
            self, sequence1, sequence_length1, sequence2, sequence_length2);
        if (ret != 0) {
            return NULL;
        }
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(DedupEstimator_add_sequence__doc__,
             "add_sequence($self, sequence, /)\n"
             "--\n"
             "\n"
             "Add a sequence to the deduplication estimator. \n"
             "\n"
             "  sequence\n"
             "    An ASCII string.\n");

#define DedupEstimator_add_sequence_method METH_O

static PyObject *
DedupEstimator_add_sequence(DedupEstimator *self, PyObject *sequence)
{
    if (!PyUnicode_CheckExact(sequence)) {
        PyErr_Format(PyExc_TypeError, "sequence should be a str object, got %R",
                     Py_TYPE((PyObject *)sequence));
        return NULL;
    }
    Py_ssize_t original_length = PyUnicode_GetLength(sequence);
    Py_ssize_t sequence_length = 0;
    const uint8_t *sequence_ptr =
        (const uint8_t *)PyUnicode_AsUTF8AndSize(sequence, &sequence_length);
    if (sequence_length != original_length) {
        PyErr_SetString(PyExc_ValueError,
                        "sequence should consist only of ASCII characters.");
        return NULL;
    }
    if (DedupEstimator_add_sequence_ptr(self, sequence_ptr, sequence_length) != 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(DedupEstimator_add_sequence_pair__doc__,
             "add__paired_sequence($self, sequence1, sequence2, /)\n"
             "--\n"
             "\n"
             "Add a paired sequence to the deduplication estimator. \n"
             "\n"
             "  sequence1\n"
             "    An ASCII string.\n"
             "  sequence2\n"
             "    An ASCII string.\n");

#define DedupEstimator_add_sequence_pair_method METH_VARARGS

static PyObject *
DedupEstimator_add_sequence_pair(DedupEstimator *self, PyObject *args)
{
    PyObject *sequence1_obj = NULL;
    PyObject *sequence2_obj = NULL;
    if (!PyArg_ParseTuple(args, "UU|:add_sequence_pair", &sequence1_obj,
                          &sequence2_obj)) {
        return NULL;
    }
    Py_ssize_t sequence1_length = PyUnicode_GetLength(sequence1_obj);
    Py_ssize_t sequence2_length = PyUnicode_GetLength(sequence2_obj);
    Py_ssize_t utf8_length1;
    Py_ssize_t utf8_length2;
    const uint8_t *sequence1 =
        (const uint8_t *)PyUnicode_AsUTF8AndSize(sequence1_obj, &utf8_length1);
    const uint8_t *sequence2 =
        (const uint8_t *)PyUnicode_AsUTF8AndSize(sequence2_obj, &utf8_length2);
    if (sequence1_length != utf8_length1) {
        PyErr_SetString(PyExc_ValueError,
                        "sequence should consist only of ASCII characters.");
        return NULL;
    }
    if (sequence2_length != utf8_length2) {
        PyErr_SetString(PyExc_ValueError,
                        "sequence should consist only of ASCII characters.");
        return NULL;
    }
    if (DedupEstimator_add_sequence_pair_ptr(self, sequence1, sequence1_length,
                                             sequence2, sequence2_length) != 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(DedupEstimator_duplication_counts__doc__,
             "duplication_counts($self)\n"
             "--\n"
             "\n"
             "Return a array.array with only the counts. \n");

#define DedupEstimator_duplication_counts_method METH_NOARGS

static PyObject *
DedupEstimator_duplication_counts(DedupEstimator *self,
                                  PyObject *Py_UNUSED(ignore))
{
    struct QCModuleState *state = get_qc_module_state_from_obj(self);
    if (state == NULL) {
        return NULL;
    }
    size_t tracked_sequences = self->stored_entries;
    uint64_t *counts = PyMem_Calloc(tracked_sequences, sizeof(uint64_t));
    if (counts == NULL) {
        return PyErr_NoMemory();
    }
    struct EstimatorEntry *hash_table = self->hash_table;
    size_t hash_table_size = self->hash_table_size;
    size_t count_index = 0;
    for (size_t i = 0; i < hash_table_size; i++) {
        struct EstimatorEntry *entry = hash_table + i;
        uint64_t count = entry->count;
        if (count == 0) {
            continue;
        }
        counts[count_index] = count;
        count_index += 1;
    }
    PyObject *result =
        PythonArray_FromBuffer('Q', counts, tracked_sequences * sizeof(uint64_t),
                               state->PythonArray_Type);
    PyMem_Free(counts);
    return result;
}

static PyMethodDef DedupEstimator_methods[] = {
    {"add_record_array", (PyCFunction)DedupEstimator_add_record_array,
     DedupEstimator_add_record_array_method,
     DedupEstimator_add_record_array__doc__},
    {"add_record_array_pair", (PyCFunction)DedupEstimator_add_record_array_pair,
     DedupEstimator_add_record_array_pair_method,
     DedupEstimator_add_record_array_pair__doc__},
    {"add_sequence", (PyCFunction)DedupEstimator_add_sequence,
     DedupEstimator_add_sequence_method, DedupEstimator_add_sequence__doc__},
    {"add_sequence_pair", (PyCFunction)DedupEstimator_add_sequence_pair,
     DedupEstimator_add_sequence_pair_method,
     DedupEstimator_add_sequence_pair__doc__},
    {"duplication_counts", (PyCFunction)DedupEstimator_duplication_counts,
     DedupEstimator_duplication_counts_method,
     DedupEstimator_duplication_counts__doc__},
    {NULL},
};

static PyMemberDef DedupEstimator_members[] = {
    {"_modulo_bits", T_ULONGLONG, offsetof(DedupEstimator, modulo_bits),
     READONLY, NULL},
    {"_hash_table_size", T_ULONGLONG,
     offsetof(DedupEstimator, hash_table_size), READONLY, NULL},
    {"tracked_sequences", T_ULONGLONG,
     offsetof(DedupEstimator, stored_entries), READONLY, NULL},
    {"front_sequence_length", T_ULONGLONG,
     offsetof(DedupEstimator, front_sequence_length), READONLY, NULL},
    {"back_sequence_length", T_ULONGLONG,
     offsetof(DedupEstimator, back_sequence_length), READONLY, NULL},
    {"front_sequence_offset", T_ULONGLONG,
     offsetof(DedupEstimator, front_sequence_offset), READONLY, NULL},
    {"back_sequence_offset", T_ULONGLONG,
     offsetof(DedupEstimator, back_sequence_offset), READONLY, NULL},
    {NULL},
};

static PyType_Slot DedupEstimator_slots[] = {
    {Py_tp_dealloc, (destructor)DedupEstimator_dealloc},
    {Py_tp_new, (newfunc)DedupEstimator__new__},
    {Py_tp_methods, DedupEstimator_methods},
    {Py_tp_members, DedupEstimator_members},
    {0, NULL},
};

static PyType_Spec DedupEstimator_spec = {
    .name = "_qc.DedupEstimator",
    .basicsize = sizeof(DedupEstimator),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT,
    .slots = DedupEstimator_slots,
};

/********************
 * NANOSTATS MODULE *
 *********************/

struct NanoInfo {
    time_t start_time;
    float duration;
    int32_t channel_id;
    uint32_t length;
    double cumulative_error_rate;
    uint64_t parent_id_hash;
};

typedef struct {
    PyObject_HEAD
    struct NanoInfo info;
} NanoporeReadInfo;

static PyObject *
NanoporeReadInfo_get_start_time(NanoporeReadInfo *self, void *closure)
{
    return PyLong_FromLong(self->info.start_time);
}
static PyObject *
NanoporeReadInfo_get_channel_id(NanoporeReadInfo *self, void *closure)
{
    return PyLong_FromLong(self->info.channel_id);
}
static PyObject *
NanoporeReadInfo_get_length(NanoporeReadInfo *self, void *closure)
{
    return PyLong_FromUnsignedLong(self->info.length);
}
static PyObject *
NanoporeReadInfo_get_cumulative_error_rate(NanoporeReadInfo *self, void *closure)
{
    return PyFloat_FromDouble(self->info.cumulative_error_rate);
}
static PyObject *
NanoporeReadInfo_get_duration(NanoporeReadInfo *self, void *closure)
{
    return PyFloat_FromDouble((double)self->info.duration);
}

static PyObject *
NanoporeReadInfo_get_parent_id_hash(NanoporeReadInfo *self, void *closure)
{
    return PyLong_FromUnsignedLongLong(self->info.parent_id_hash);
}

static PyGetSetDef NanoporeReadInfo_properties[] = {
    {"start_time", (getter)NanoporeReadInfo_get_start_time, NULL,
     "unix UTC timestamp for start time", NULL},
    {"channel_id", (getter)NanoporeReadInfo_get_channel_id, NULL,
     "channel number", NULL},
    {"length", (getter)NanoporeReadInfo_get_length, NULL, NULL, NULL},
    {"cumulative_error_rate", (getter)NanoporeReadInfo_get_cumulative_error_rate,
     NULL, "sum off all the bases' error rates.", NULL},
    {"duration", (getter)NanoporeReadInfo_get_duration, NULL, NULL, NULL},
    {"parent_id_hash", (getter)NanoporeReadInfo_get_parent_id_hash, NULL, NULL,
     NULL},
    {NULL},
};

static PyType_Slot NanoporeReadInfo_slots[] = {
    {Py_tp_dealloc, (destructor)PyObject_DEL},
    {Py_tp_getset, NanoporeReadInfo_properties},
    {0, NULL},
};

static PyType_Spec NanoporeReadInfo_spec = {
    .name = "_qc.NanoporeReadInfo",
    .basicsize = sizeof(NanoporeReadInfo),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT,
    .slots = NanoporeReadInfo_slots,
};

typedef struct _NanoStatsStruct {
    PyObject_HEAD
    bool skipped;
    size_t number_of_reads;
    size_t nano_infos_size;
    struct NanoInfo *nano_infos;
    time_t min_time;
    time_t max_time;
    PyObject *skipped_reason;
} NanoStats;

static void
NanoStats_dealloc(NanoStats *self)
{
    PyMem_Free(self->nano_infos);
    Py_XDECREF(self->skipped_reason);
    PyTypeObject *tp = Py_TYPE((PyObject *)self);
    PyObject_FREE(self);
    Py_XDECREF((PyObject *)tp);
}

typedef struct {
    PyObject_HEAD
    size_t number_of_reads;
    struct NanoInfo *nano_infos;
    size_t current_pos;
    PyObject *nano_stats;
    PyTypeObject *NanoporeReadInfo_Type;
} NanoStatsIterator;

static void
NanoStatsIterator_dealloc(NanoStatsIterator *self)
{
    Py_XDECREF(self->nano_stats);
    Py_XDECREF((PyObject *)self->NanoporeReadInfo_Type);
    PyTypeObject *tp = Py_TYPE((PyObject *)self);
    PyObject_Free(self);
    Py_XDECREF((PyObject *)tp);
}

static PyObject *
NanoStatsIterator_FromNanoStats(NanoStats *nano_stats)
{
    struct QCModuleState *state =
        get_qc_module_state_from_obj((PyObject *)nano_stats);
    NanoStatsIterator *self =
        PyObject_New(NanoStatsIterator, state->NanoStatsIterator_Type);
    if (self == NULL) {
        return PyErr_NoMemory();
    }
    self->NanoporeReadInfo_Type =
        (PyTypeObject *)Py_NewRef(state->NanoporeReadInfo_Type);
    self->nano_infos = nano_stats->nano_infos;
    self->number_of_reads = nano_stats->number_of_reads;
    self->current_pos = 0;
    Py_INCREF((PyObject *)nano_stats);
    self->nano_stats = (PyObject *)nano_stats;
    return (PyObject *)self;
}

static PyObject *
NanoStatsIterator__iter__(NanoStatsIterator *self)
{
    Py_INCREF((PyObject *)self);
    return (PyObject *)self;
}

static PyObject *
NanoStatsIterator__next__(NanoStatsIterator *self)
{
    size_t current_pos = self->current_pos;
    if (current_pos == self->number_of_reads) {
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }
    NanoporeReadInfo *info =
        PyObject_New(NanoporeReadInfo, self->NanoporeReadInfo_Type);
    if (info == NULL) {
        return PyErr_NoMemory();
    }
    memcpy(&info->info, self->nano_infos + current_pos, sizeof(struct NanoInfo));
    self->current_pos = current_pos + 1;
    return (PyObject *)info;
}

static PyType_Slot NanoStatsIterator_slots[] = {
    {Py_tp_dealloc, (destructor)NanoStatsIterator_dealloc},
    {Py_tp_iter, (iternextfunc)NanoStatsIterator__iter__},
    {Py_tp_iternext, (iternextfunc)NanoStatsIterator__next__},
    {0, NULL},
};

static PyType_Spec NanoStatsIterator_spec = {
    .name = "_qc.NanoStatsIterator",
    .basicsize = sizeof(NanoStatsIterator),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT,
    .slots = NanoStatsIterator_slots,
};

static PyObject *
NanoStats__new__(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    static char *format = {":_qc.NanoStats"};
    static char *kwarg_names[] = {NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, kwarg_names)) {
        return NULL;
    }
    NanoStats *self = PyObject_New(NanoStats, type);
    if (self == NULL) {
        return PyErr_NoMemory();
    }
    self->nano_infos = NULL;
    self->nano_infos_size = 0;
    self->number_of_reads = 0;
    self->skipped = false;
    self->skipped_reason = NULL;
    self->min_time = 0;
    self->max_time = 0;
    return (PyObject *)self;
}

/**
 * @brief Parse read, channel_id and start_time fields from a nanopore FASTQ
 *        header. Other fields are untouched.
 *
 * @param header The header.
 * @param header_length size of the header.
 * @param info Pointer to the info object to be populated.
 * @return int 0 on success, -1 on parsing error.
 */
static int
NanoInfo_from_header(const uint8_t *header, size_t header_length,
                     struct NanoInfo *info)
{
    const uint8_t *cursor = header;
    const uint8_t *end_ptr = header + header_length;
    cursor = memchr(cursor, ' ', header_length);
    if (cursor == NULL) {
        return -1;
    }
    cursor += 1;
    int32_t channel_id = -1;
    time_t start_time = -1;
    while (cursor < end_ptr) {
        const uint8_t *field_name = cursor;
        const uint8_t *equals = memchr(field_name, '=', end_ptr - field_name);
        if (equals == NULL) {
            return -1;
        }
        size_t field_name_length = equals - field_name;
        const uint8_t *field_value = equals + 1;
        const uint8_t *field_end =
            memchr(field_value, ' ', end_ptr - field_value);
        if (field_end == NULL) {
            field_end = end_ptr;
        }
        cursor = field_end + 1;
        switch (field_name_length) {
            case 2:
                if (memcmp(field_name, "ch", 2) == 0) {
                    channel_id = unsigned_decimal_integer_from_string(
                        field_value, field_end - field_value);
                }
                break;
            case 10:
                if (memcmp(field_name, "start_time", 10) == 0) {
                    start_time = time_string_to_timestamp(field_value);
                }
                break;
        }
    }
    if (channel_id == -1 || start_time == -1) {
        return -1;
    }
    info->channel_id = channel_id;
    info->start_time = start_time;
    return 0;
}

static Py_ssize_t
get_tag_int_value(const uint8_t *tag)
{
    uint8_t tag_type = tag[2];
    const uint8_t *value_start = tag + 3;
    switch (tag_type) {
        case 'c':
            return ((int8_t *)value_start)[0];
        case 'C':
            return ((uint8_t *)value_start)[0];
        case 's':
            return ((int16_t *)value_start)[0];
        case 'S':
            return ((uint16_t *)value_start)[0];
        case 'i':
            return ((int32_t *)value_start)[0];
        case 'I':
            return ((uint32_t *)value_start)[0];
        default:
            return PY_SSIZE_T_MIN;
    }
}

static Py_ssize_t
tag_length(const uint8_t *tag, size_t maximum_tag_length)
{
    if (maximum_tag_length < 4) {
        PyErr_SetString(PyExc_ValueError, "truncated tags");
        return -1;
    }
    uint8_t tag_type = tag[2];
    const uint8_t *value_start = tag + 3;
    size_t value_length;
    bool is_array = false;
    uint32_t array_length = 1;
    if (tag_type == 'B') {
        is_array = true;
        value_start = tag + 8;
        tag_type = tag[3];
        if (maximum_tag_length < 8) {
            PyErr_SetString(PyExc_ValueError, "truncated tags");
            return -1;
        }
        array_length = *(uint32_t *)(tag + 4);
    };

    switch (tag_type) {
        case 'A':
        case 'c':
        case 'C':
            value_length = 1;
            break;
        case 's':
        case 'S':
            value_length = 2;
            break;
        case 'I':
        case 'i':
        case 'f':
            value_length = 4;
            break;
        case 'Z':
        case 'H':
            if (is_array) {
                PyErr_Format(PyExc_ValueError, "Invalid type for array %c",
                             tag_type);
                return -1;
            }
            uint8_t *string_end = memchr(value_start, 0, maximum_tag_length - 3);
            if (string_end == NULL) {
                PyErr_SetString(PyExc_ValueError, "truncated tags");
                return -1;
            }
            value_length =
                (string_end - value_start) + 1;  // +1 for terminating null
            break;
        default:
            PyErr_Format(PyExc_ValueError, "Unknown tag type %c", tag_type);
            return -1;
    }
    size_t this_tag_length = (value_start - tag) + array_length * value_length;
    if (this_tag_length > maximum_tag_length) {
        PyErr_SetString(PyExc_ValueError, "truncated tags");
        return -1;
    }
    return this_tag_length;
}

struct TagInfo {
    int32_t channel_id;
    float duration;
    time_t start_time;
    uint64_t parent_id_hash;
};

/**
 * @brief "Hash" a uuid4 by using the first 8 digits and last 8 digits for
 * 64 random bits. Return 0 on error.
 */
static uint64_t
uuid4_hash(char *uuid)
{
    /* UUID4 takes the form of,
        xxxxxxxx-xxxx-Mxxx-Nxxx-xxxxxxxxxxxx
        ^^^^^^^^                    ^^^^^^^^
        These hexadecimal digits are used, 16 in total. M should be 4. N can
        be 8,9,A,B,C,D. The rest of the digits are andom.
     */
    if (uuid[8] != '-' || uuid[13] != '-' ||
        uuid[14] != '4' ||  // UUID version 4 check.
        uuid[18] != '-' || uuid[23] != '-' || uuid[36] != 0) {
        return 0;
    }
    char *end_ptr = uuid;
    uint64_t first_bit = strtoull(uuid, &end_ptr, 16);
    if (end_ptr - uuid != 8) {
        // strtoull stops at first non-hexadecimal. This should be at position 8.
        return 0;
    }
    uint64_t last_bit = strtoull(uuid + 28, &end_ptr, 16);
    if (end_ptr - uuid != 36) {
        // first non-hexadecimal should be at string end.
        return 0;
    }
    return (first_bit << 32) | (last_bit & 0xFFFFFFFFULL);
}

/**
 * @brief Throw a Python RuntimeError for an unexpected typecode and return -1
 */
static inline int
tag_wrong_typecode(char *tag, char expected_typecode, char actual_typecode)
{
    PyErr_Format(PyExc_RuntimeError,
                 "Wrong tag type for '%s' expected '%c' got '%c'", tag,
                 expected_typecode, actual_typecode);
    return -1;
}

/**
 * @brief correct memcmp shorthand for ease of writing.
 */
static inline bool
has_tag_id(const uint8_t *restrict tag, char *expected_tag)
{
    return memcmp(tag, expected_tag, strlen(expected_tag)) == 0;
    ;
}

static int
TagInfo_from_tags(const uint8_t *tags, size_t tags_length, struct TagInfo *info)
{
    info->channel_id = -1;
    info->duration = 0.0;
    info->start_time = 0;
    info->parent_id_hash = 0;
    while (tags_length > 0) {
        Py_ssize_t this_tag_length = tag_length(tags, tags_length);
        if (this_tag_length == -1) {
            return -1;
        }
        const uint8_t *tag = tags;
        uint8_t typecode = tags[2];

        if (has_tag_id(tag, "ch")) {
            Py_ssize_t channel_id = get_tag_int_value(tag);
            if (channel_id == PY_SSIZE_T_MIN) {
                return -1;
            }
            info->channel_id = channel_id;
        }
        else if (has_tag_id(tag, "st")) {
            if (typecode != 'Z') {
                return tag_wrong_typecode("st", 'Z', typecode);
            }
            info->start_time = time_string_to_timestamp(tags + 3);
        }
        else if (has_tag_id(tag, "du")) {
            if (typecode != 'f') {
                return tag_wrong_typecode("du", 'f', typecode);
            }
            info->duration = ((float *)(tags + 3))[0];
        }
        else if (has_tag_id(tag, "pi")) {
            if (typecode != 'Z') {
                return tag_wrong_typecode("pi", 'Z', typecode);
            }
            const uint8_t *value = tag + 3;
            // -3 for tag id, typecode. -1 for terminating 0.
            size_t value_length = this_tag_length - 4;
            if (value_length != 36) {
                PyErr_WarnFormat(
                    PyExc_UserWarning, 1,
                    "pi tag should have a valid uuid4 format with 36 "
                    "characters. Counted %zu. Skipping tag.",
                    value_length);
            }
            else {
                info->parent_id_hash = uuid4_hash((char *)value);
            }
        }
        tags = tags + this_tag_length;
        tags_length -= this_tag_length;
    }
    return 0;
}

/**
 * @brief Add a FASTQ record to the NanoStats module
 *
 * @param self
 * @param meta
 * @return int 0 on success or when not a nanopore FASTQ. -1 on error.
 */
static int
NanoStats_add_meta(NanoStats *self, struct FastqMeta *meta)
{
    if (self->skipped) {
        return 0;
    }
    if (self->number_of_reads == self->nano_infos_size) {
        size_t old_size = self->nano_infos_size;
        size_t new_size = Py_MAX(old_size * 2, 16 * 1024);
        struct NanoInfo *tmp =
            PyMem_Realloc(self->nano_infos, new_size * sizeof(struct NanoInfo));
        if (tmp == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        memset(tmp + old_size, 0, (new_size - old_size) * sizeof(struct NanoInfo));
        self->nano_infos = tmp;
        self->nano_infos_size = new_size;
    };
    struct NanoInfo *info = self->nano_infos + self->number_of_reads;
    size_t sequence_length = meta->sequence_length;
    info->length = sequence_length;

    if (meta->tags_length) {
        struct TagInfo tag_info;
        if (TagInfo_from_tags(meta->record_start + meta->tags_offset,
                              meta->tags_length, &tag_info) != 0) {
            return -1;
        }
        info->channel_id = tag_info.channel_id;
        info->duration = tag_info.duration;
        info->start_time = tag_info.start_time;
        info->parent_id_hash = tag_info.parent_id_hash;
    }
    else if (NanoInfo_from_header(meta->name, meta->name_length, info) != 0) {
        PyObject *header_obj = PyUnicode_DecodeASCII((const char *)meta->name,
                                                     meta->name_length, NULL);
        if (header_obj == NULL) {
            return -1;
        }
        self->skipped = true;
        self->skipped_reason =
            PyUnicode_FromFormat("Can not parse header: %R", header_obj);
        Py_DECREF(header_obj);
        return 0;
    }
    info->cumulative_error_rate = meta->accumulated_error_rate;
    time_t timestamp = info->start_time;
    if (timestamp > self->max_time) {
        self->max_time = timestamp;
    }
    if (self->min_time == 0 || timestamp < self->min_time) {
        self->min_time = timestamp;
    }
    self->number_of_reads += 1;
    return 0;
}

PyDoc_STRVAR(NanoStats_add_read__doc__,
             "add_read($self, read, /)\n"
             "--\n"
             "\n"
             "Add a read to the NanoStats module. \n"
             "\n"
             "  read\n"
             "    A FastqRecordView object.\n");

#define NanoStats_add_read_method METH_O

static PyObject *
NanoStats_add_read(NanoStats *self, FastqRecordView *read)
{
    int is_view = is_FastqRecordView(self, read);
    if (is_view == -1) {
        return NULL;
    }
    else if (is_view == 0) {
        PyErr_Format(PyExc_TypeError,
                     "read should be a FastqRecordView object, got %R",
                     Py_TYPE((PyObject *)read));
        return NULL;
    }
    if (NanoStats_add_meta(self, &read->meta) != 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(NanoStats_add_record_array__doc__,
             "add_record_array($self, record_array, /)\n"
             "--\n"
             "\n"
             "Add a record_array to the NanoStats module. \n"
             "\n"
             "  record_array\n"
             "    A FastqRecordArrayView object.\n");

#define NanoStats_add_record_array_method METH_O

static PyObject *
NanoStats_add_record_array(NanoStats *self, FastqRecordArrayView *record_array)
{
    int is_record_array = is_FastqRecordArrayView(self, record_array);
    if (is_record_array == -1) {
        return NULL;
    }
    else if (is_record_array == 0) {
        PyErr_Format(
            PyExc_TypeError,
            "record_array should be a FastqRecordArrayView object, got %R",
            Py_TYPE((PyObject *)record_array));
        return NULL;
    }
    if (self->skipped) {
        Py_RETURN_NONE;
    }
    Py_ssize_t number_of_records = Py_SIZE((PyObject *)record_array);
    struct FastqMeta *records = record_array->records;
    for (Py_ssize_t i = 0; i < number_of_records; i++) {
        if (NanoStats_add_meta(self, records + i) != 0) {
            return NULL;
        }
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(NanoStats_nano_info_iterator__doc__,
             "nano_info_iterator($self, /)\n"
             "--\n"
             "\n"
             "Return an iterator of NanoporeReadInfo objects. \n");

#define NanoStats_nano_info_iterator_method METH_NOARGS

static PyObject *
NanoStats_nano_info_iterator(NanoStats *self, PyObject *Py_UNUSED(ignore))
{
    return NanoStatsIterator_FromNanoStats(self);
}

static PyMethodDef NanoStats_methods[] = {
    {"add_read", (PyCFunction)NanoStats_add_read, NanoStats_add_read_method,
     NanoStats_add_read__doc__},
    {"add_record_array", (PyCFunction)NanoStats_add_record_array,
     NanoStats_add_record_array_method, NanoStats_add_record_array__doc__},
    {"nano_info_iterator", (PyCFunction)NanoStats_nano_info_iterator,
     NanoStats_nano_info_iterator_method, NanoStats_nano_info_iterator__doc__},
    {NULL},
};

static PyMemberDef NanoStats_members[] = {
    {"number_of_reads", T_ULONGLONG, offsetof(NanoStats, number_of_reads),
     READONLY, "The total amount of reads counted"},
    {"skipped_reason", T_OBJECT, offsetof(NanoStats, skipped_reason), READONLY,
     "What the reason is for skipping the module if skipped."
     "Set to None if not skipped."},
    {
        "minimum_time",
        T_LONG,
        offsetof(NanoStats, min_time),
        READONLY,
        "The earliest timepoint found in the headers",
    },
    {"maximum_time", T_LONG, offsetof(NanoStats, max_time), READONLY,
     "The latest timepoint found in the headers"},
    {NULL},
};

static PyType_Slot NanoStats_slots[] = {
    {Py_tp_dealloc, (destructor)NanoStats_dealloc},
    {Py_tp_new, (newfunc)NanoStats__new__},
    {Py_tp_methods, NanoStats_methods},
    {Py_tp_members, NanoStats_members},
    {0, NULL},
};

static PyType_Spec NanoStats_spec = {
    .name = "_qc.NanoStats",
    .basicsize = sizeof(NanoStats),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT,
    .slots = NanoStats_slots,
};

/***********************
 * INSERT SIZE METRICS *
 ***********************/

#define INSERT_SIZE_MAX_ADAPTERS 10000
#define INSERT_SIZE_MAX_ADAPTER_STORE_SIZE 31

struct AdapterTableEntry {
    uint64_t hash;
    uint64_t adapter_count;
    uint8_t adapter_length;
    uint8_t adapter[INSERT_SIZE_MAX_ADAPTER_STORE_SIZE];
};

typedef struct _InsertSizeMetricsStruct {
    PyObject_HEAD
    uint64_t *insert_sizes;
    uint64_t total_reads;
    uint64_t number_of_adapters_read1;
    uint64_t number_of_adapters_read2;
    struct AdapterTableEntry *hash_table_read1;
    struct AdapterTableEntry *hash_table_read2;
    size_t max_adapters;
    size_t hash_table_size;
    size_t hash_table_read1_entries;
    size_t hash_table_read2_entries;
    size_t max_insert_size;
} InsertSizeMetrics;

static void
InsertSizeMetrics_dealloc(InsertSizeMetrics *self)
{
    PyMem_Free(self->hash_table_read1);
    PyMem_Free(self->hash_table_read2);
    PyMem_Free(self->insert_sizes);
    PyTypeObject *tp = Py_TYPE((PyObject *)self);
    PyObject_Free(self);
    Py_XDECREF((PyObject *)tp);
}

static PyMemberDef InsertSizeMetrics_members[] = {
    {"total_reads", T_ULONGLONG, offsetof(InsertSizeMetrics, total_reads),
     READONLY, "the total number of reads"},
    {"number_of_adapters_read1", T_ULONGLONG,
     offsetof(InsertSizeMetrics, number_of_adapters_read1), READONLY,
     "The number off reads in read 1 with an adapter."},
    {"number_of_adapters_read2", T_ULONGLONG,
     offsetof(InsertSizeMetrics, number_of_adapters_read2), READONLY,
     "The number off reads in read 2 with an adapter."},
    {NULL},
};

static PyObject *
InsertSizeMetrics__new__(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    Py_ssize_t max_adapters = INSERT_SIZE_MAX_ADAPTERS;
    static char *format = "|n:InsertSizeMetrics.__new__";
    static char *keywords[] = {"max_adapters", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, keywords,
                                     &max_adapters)) {
        return NULL;
    }

    if (max_adapters < 1) {
        PyErr_Format(PyExc_ValueError,
                     "max_adapters must be at least 1, got %zd", max_adapters);
        return NULL;
    }

    InsertSizeMetrics *self = PyObject_New(InsertSizeMetrics, type);
    if (self == NULL) {
        return PyErr_NoMemory();
    }
    size_t hash_table_bits = (size_t)(log2(max_adapters * 1.5) + 1);

    self->max_adapters = max_adapters;
    self->max_insert_size = 0;
    self->hash_table_read1_entries = 0;
    self->hash_table_read2_entries = 0;
    self->hash_table_size = 1 << hash_table_bits;
    self->hash_table_read1 =
        PyMem_Calloc(self->hash_table_size, sizeof(struct AdapterTableEntry));
    self->hash_table_read2 =
        PyMem_Calloc(self->hash_table_size, sizeof(struct AdapterTableEntry));
    self->insert_sizes =
        PyMem_Calloc(self->max_insert_size + 1, sizeof(uint64_t));
    self->total_reads = 0;
    self->number_of_adapters_read1 = 0;
    self->number_of_adapters_read2 = 0;

    if (self->hash_table_read1 == NULL || self->hash_table_read2 == NULL ||
        self->insert_sizes == NULL) {
        /* Memory gets freed in the dealloc method. */
        Py_DECREF(self);
        return PyErr_NoMemory();
    }
    return (PyObject *)self;
}

static int
InsertSizeMetrics_resize(InsertSizeMetrics *self, size_t new_size)
{
    if (new_size <= self->max_insert_size) {
        return 0;
    }
    size_t old_size = self->max_insert_size;
    size_t new_raw_size = sizeof(uint64_t) * (new_size + 1);
    uint64_t *tmp = PyMem_Realloc(self->insert_sizes, new_raw_size);
    if (tmp == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    memset(tmp + old_size + 1, 0, (new_size - old_size) * sizeof(uint64_t));
    self->max_insert_size = new_size;
    self->insert_sizes = tmp;
    return 0;
}

static inline void
InsertSizeMetrics_add_adapter(InsertSizeMetrics *self, const uint8_t *adapter,
                              size_t adapter_length, bool read2)
{
    assert(adapter_length <= INSERT_SIZE_MAX_ADAPTER_STORE_SIZE);
    uint64_t hash = MurmurHash3_x64_64(adapter, adapter_length, 0);
    size_t hash_table_size = self->hash_table_size;
    struct AdapterTableEntry *hash_table = self->hash_table_read1;
    size_t *current_entries = &self->hash_table_read1_entries;
    if (read2) {
        hash_table = self->hash_table_read2;
        current_entries = &self->hash_table_read2_entries;
    }
    bool hash_table_full = *current_entries == self->max_adapters;

    size_t hash_to_index_int =
        hash_table_size - 1;  // Works because size is a power of 2.
    size_t index = hash & hash_to_index_int;
    while (true) {
        struct AdapterTableEntry *entry = hash_table + index;
        uint64_t current_hash = entry->hash;
        if (current_hash == hash) {
            if (adapter_length == entry->adapter_length &&
                memcmp(adapter, entry->adapter, adapter_length) == 0) {
                entry->adapter_count += 1;
                return;
            }
        }
        else if (entry->adapter_count == 0) {
            if (!hash_table_full) {
                entry->hash = hash;
                entry->adapter_length = adapter_length;
                memcpy(entry->adapter, adapter, adapter_length);
                entry->adapter_count = 1;
                current_entries[0] += 1;
            }
            return;
        }
        index += 1;
        index &= hash_to_index_int;
    }
}

// clang-format off
static const uint8_t NUCLEOTIDE_COMPLEMENT[128] = {
// All non-ACGT become 0 so they don't match with N.
// Control characters
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// Interpunction numbers etc
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//      A,  B,  C,  D, E, F,  G,  H, I, J, K, L, M, N, O,
    0, 'T', 0, 'G', 0, 0, 0, 'C', 0, 0, 0, 0, 0, 0, 0, 0,
//  P, Q, R, S,  T,  U,  V, W, X, Y, Z,  
    0, 0, 0, 0, 'A', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//      a,  b,  c,  d, e, f,  g,  h, i, j, k, l, m, n, o,
    0, 'T', 0, 'G', 0, 0, 0, 'C', 0, 0, 0, 0, 0, 0, 0, 0,
//  p, q, r, s,  t,  u,  v, w, x, y, z, 
    0, 0, 0, 0, 'A', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
};
// clang-format on

static inline void
reverse_complement(uint8_t *restrict dest, const uint8_t *restrict src,
                   size_t length)
{
    size_t dest_index = length;
    for (size_t src_index = 0; src_index < length; src_index++) {
        dest_index -= 1;
        dest[dest_index] = NUCLEOTIDE_COMPLEMENT[src[src_index]];
    }
}

static inline size_t
hamming_distance(const uint8_t *restrict sequence1,
                 const uint8_t *restrict sequence2, size_t length)
{
    size_t distance = 0;
    for (size_t i = 0; i < length; i++) {
        if (sequence1[i] != sequence2[i]) {
            distance += 1;
        }
    }
    return distance;
}

/* Everything set except the bit for 32. This is the difference in ASCII
   between lowercase and uppercase. */

#define UPPER_MASK 0xDFDFDFDFDFDFDFDFULL

/**
 * @brief Determine insert size between sequences by calculating the overlap.
 *
 * @return Py_ssize_t 0, when no overlap could be determined.
 */
static size_t
calculate_insert_size(const uint8_t *restrict sequence1, size_t sequence1_length,
                      const uint8_t *restrict sequence2, size_t sequence2_length)
{
    /* The needle size is 16. One error is allowed. By hardcoding is it can
       be optimized by looking for 2 64-bit integers instead. At least one of
       the 64-bit integers must find a match at a position if there is only one
       error. This is the pigeon hole principle. This way the sequence can be
       searched quickly, while allowing errors. */

    if (sequence2_length < 16 || sequence1_length < 16) {
        return 0;
    }
    uint8_t seq_store[32];
    uint8_t *start_seq = seq_store;
    uint8_t *end_seq = ((uint8_t *)seq_store) + 16;
    reverse_complement(start_seq, sequence2, 16);
    reverse_complement(end_seq, sequence2 + sequence2_length - 16, 16);

    uint64_t start1 = ((uint64_t *)start_seq)[0];
    uint64_t start2 = ((uint64_t *)start_seq)[1];
    uint64_t end1 = ((uint64_t *)end_seq)[0];
    uint64_t end2 = ((uint64_t *)end_seq)[1];

    size_t run_length = sequence1_length - 15;
    for (size_t i = 0; i < run_length; i++) {
        uint64_t word1 = ((uint64_t *)(sequence1 + i))[0] & UPPER_MASK;
        uint64_t word2 = ((uint64_t *)(sequence1 + i))[1] & UPPER_MASK;
        if (start1 == word1 || start2 == word2) {
            if (hamming_distance(sequence1 + i, start_seq, 16) <= 1) {
                return i + 16;
            }
        }
        if (end1 == word1 || end2 == word2) {
            if (hamming_distance(sequence1 + i, end_seq, 16) <= 1) {
                return i + sequence2_length;
            }
        }
    }
    return 0;  // No matches found.
}

static int
InsertSizeMetrics_add_sequence_pair_ptr(InsertSizeMetrics *self,
                                        const uint8_t *sequence1,
                                        size_t sequence1_length,
                                        const uint8_t *sequence2,
                                        size_t sequence2_length)
{
    size_t insert_size = calculate_insert_size(sequence1, sequence1_length,
                                               sequence2, sequence2_length);
    if (insert_size > self->max_insert_size) {
        if (InsertSizeMetrics_resize(self, insert_size) != 0) {
            return -1;
        };
    }
    self->total_reads += 1;
    self->insert_sizes[insert_size] += 1;
    /* Don't store adapters when no overlap is detected. */
    if ((insert_size) == 0) {
        return 0;
    }
    Py_ssize_t remainder1 = (Py_ssize_t)sequence1_length - (Py_ssize_t)insert_size;
    if (remainder1 > 0) {
        self->number_of_adapters_read1 += 1;
        InsertSizeMetrics_add_adapter(
            self, sequence1 + insert_size,
            Py_MIN(remainder1, INSERT_SIZE_MAX_ADAPTER_STORE_SIZE), false);
    }
    Py_ssize_t remainder2 = (Py_ssize_t)sequence2_length - (Py_ssize_t)insert_size;
    if (remainder2 > 0) {
        self->number_of_adapters_read2 += 1;
        InsertSizeMetrics_add_adapter(
            self, sequence2 + insert_size,
            Py_MIN(remainder2, INSERT_SIZE_MAX_ADAPTER_STORE_SIZE), true);
    }
    return 0;
}

PyDoc_STRVAR(InsertSizeMetrics_add_sequence_pair__doc__,
             "add_sequence_pair($self, sequence1, sequence2, /)\n"
             "--\n"
             "\n"
             "Add a paired sequence to the insert size metrics. \n"
             "\n"
             "  sequence1\n"
             "    An ASCII string.\n"
             "  sequence2\n"
             "    An ASCII string.\n");

#define InsertSizeMetrics_add_sequence_pair_method METH_VARARGS

static PyObject *
InsertSizeMetrics_add_sequence_pair(InsertSizeMetrics *self, PyObject *args)
{
    PyObject *sequence1_obj = NULL;
    PyObject *sequence2_obj = NULL;
    if (!PyArg_ParseTuple(args, "UU|:InsertSizeMetrics.add_sequence_pair",
                          &sequence1_obj, &sequence2_obj)) {
        return NULL;
    }
    Py_ssize_t sequence1_length = PyUnicode_GetLength(sequence1_obj);
    Py_ssize_t sequence2_length = PyUnicode_GetLength(sequence2_obj);
    Py_ssize_t utf8_length1;
    Py_ssize_t utf8_length2;
    const uint8_t *sequence1 =
        (const uint8_t *)PyUnicode_AsUTF8AndSize(sequence1_obj, &utf8_length1);
    const uint8_t *sequence2 =
        (const uint8_t *)PyUnicode_AsUTF8AndSize(sequence2_obj, &utf8_length2);
    if (sequence1_length != utf8_length1) {
        PyErr_SetString(PyExc_ValueError,
                        "sequence1 should consist only of ASCII characters.");
        return NULL;
    }
    if (sequence2_length != utf8_length2) {
        PyErr_SetString(PyExc_ValueError,
                        "sequence2 should consist only of ASCII characters.");
        return NULL;
    }
    int ret = InsertSizeMetrics_add_sequence_pair_ptr(
        self, sequence1, sequence1_length, sequence2, sequence2_length);
    if (ret != 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(InsertSizeMetrics_add_record_array_pair__doc__,
             "add_record_array_pair($self, record_array1, record_array2 /)\n"
             "--\n"
             "\n"
             "Add a pair of record arrays to the insert size metrics. \n"
             "\n"
             "  record_array1\n"
             "    A FastqRecordArrayView object. First of read pair.\n"
             "  record_array2\n"
             "    A FastqRecordArrayView object. Second of read pair.\n");

#define InsertSizeMetrics_add_record_array_pair_method METH_FASTCALL

static PyObject *
InsertSizeMetrics_add_record_array_pair(InsertSizeMetrics *self,
                                        PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2) {
        PyErr_Format(PyExc_TypeError,
                     "InsertSizeMetrics.add_record_array_pair() "
                     "takes exactly two arguments, got %zd",
                     nargs);
    }
    FastqRecordArrayView *record_array1 = (FastqRecordArrayView *)args[0];
    FastqRecordArrayView *record_array2 = (FastqRecordArrayView *)args[1];
    int is_record_array1 = is_FastqRecordArrayView(self, record_array1);
    if (is_record_array1 == -1) {
        return NULL;
    }
    else if (is_record_array1 == 0) {
        PyErr_Format(
            PyExc_TypeError,
            "record_array1 should be a FastqRecordArrayView object, got %R",
            Py_TYPE((PyObject *)record_array1));
        return NULL;
    }
    int is_record_array2 = is_FastqRecordArrayView(self, record_array2);
    if (is_record_array2 == -1) {
        return NULL;
    }
    else if (is_record_array2 == 0) {
        PyErr_Format(
            PyExc_TypeError,
            "record_array2 should be a FastqRecordArrayView object, got %R",
            Py_TYPE((PyObject *)record_array2));
        return NULL;
    }
    Py_ssize_t number_of_records = Py_SIZE((PyObject *)record_array1);
    if (Py_SIZE((PyObject *)record_array2) != number_of_records) {
        PyErr_Format(
            PyExc_ValueError,
            "record_array1 and record_array2 must be of the same size. "
            "Got %zd and %zd respectively.",
            number_of_records, Py_SIZE((PyObject *)record_array2));
    }
    struct FastqMeta *records1 = record_array1->records;
    struct FastqMeta *records2 = record_array2->records;
    for (Py_ssize_t i = 0; i < number_of_records; i++) {
        struct FastqMeta *meta1 = records1 + i;
        struct FastqMeta *meta2 = records2 + i;
        uint8_t *sequence1 = meta1->record_start + meta1->sequence_offset;
        uint8_t *sequence2 = meta2->record_start + meta2->sequence_offset;
        size_t sequence_length1 = meta1->sequence_length;
        size_t sequence_length2 = meta2->sequence_length;
        int ret = InsertSizeMetrics_add_sequence_pair_ptr(
            self, sequence1, sequence_length1, sequence2, sequence_length2);
        if (ret != 0) {
            return NULL;
        }
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(InsertSizeMetrics_insert_sizes__doc__,
             "insert_sizes($self)\n"
             "--\n"
             "\n"
             "Retrieve an array of the insert sizes.\n");

#define InsertSizeMetrics_insert_sizes_method METH_NOARGS

static PyObject *
InsertSizeMetrics_insert_sizes(InsertSizeMetrics *self,
                               PyObject *Py_UNUSED(ignore))
{
    struct QCModuleState *state = get_qc_module_state_from_obj(self);
    if (state == NULL) {
        return NULL;
    }
    return PythonArray_FromBuffer('Q', self->insert_sizes,
                                  (self->max_insert_size + 1) * sizeof(uint64_t),
                                  state->PythonArray_Type);
}

static PyObject *
adapter_hash_table_to_python_list(struct AdapterTableEntry *hash_table,
                                  size_t hash_table_size)
{
    PyObject *adapter_list = PyList_New(0);
    struct AdapterTableEntry *entries = hash_table;
    for (size_t i = 0; i < hash_table_size; i++) {
        struct AdapterTableEntry *entry = entries + i;
        uint64_t adapter_count = entry->adapter_count;
        if (adapter_count) {
            PyObject *adapter_tuple =
                Py_BuildValue("(s#K)", entry->adapter,
                              (Py_ssize_t)entry->adapter_length, adapter_count);
            if (adapter_tuple == NULL) {
                Py_DECREF(adapter_list);
                return NULL;
            }
            if (PyList_Append(adapter_list, adapter_tuple) != 0) {
                return NULL;
            }
            Py_DECREF(adapter_tuple);
        }
    }
    return adapter_list;
}

PyDoc_STRVAR(InsertSizeMetrics_adapters_read1__doc__,
             "adapters_read1($self)\n"
             "--\n"
             "\n"
             "Retrieve a list of adapters for read 1 with their counts.\n");

#define InsertSizeMetrics_adapters_read1_method METH_NOARGS

static PyObject *
InsertSizeMetrics_adapters_read1(InsertSizeMetrics *self,
                                 PyObject *Py_UNUSED(ignore))
{
    return adapter_hash_table_to_python_list(self->hash_table_read1,
                                             self->hash_table_size);
}

PyDoc_STRVAR(InsertSizeMetrics_adapters_read2__doc__,
             "adapters_read2($self)\n"
             "--\n"
             "\n"
             "Retrieve a list of adapters for read 2 with their counts.\n");

#define InsertSizeMetrics_adapters_read2_method METH_NOARGS

static PyObject *
InsertSizeMetrics_adapters_read2(InsertSizeMetrics *self,
                                 PyObject *Py_UNUSED(ignore))
{
    return adapter_hash_table_to_python_list(self->hash_table_read2,
                                             self->hash_table_size);
}

static PyMethodDef InsertSizeMetrics_methods[] = {
    {"add_sequence_pair", (PyCFunction)InsertSizeMetrics_add_sequence_pair,
     InsertSizeMetrics_add_sequence_pair_method,
     InsertSizeMetrics_add_sequence_pair__doc__},
    {
        "add_record_array_pair",
        (PyCFunction)InsertSizeMetrics_add_record_array_pair,
        InsertSizeMetrics_add_record_array_pair_method,
        InsertSizeMetrics_add_record_array_pair__doc__,
    },
    {"insert_sizes", (PyCFunction)InsertSizeMetrics_insert_sizes,
     InsertSizeMetrics_insert_sizes_method, InsertSizeMetrics_insert_sizes__doc__},
    {"adapters_read1", (PyCFunction)InsertSizeMetrics_adapters_read1,
     InsertSizeMetrics_adapters_read1_method,
     InsertSizeMetrics_adapters_read1__doc__},
    {"adapters_read2", (PyCFunction)InsertSizeMetrics_adapters_read2,
     InsertSizeMetrics_adapters_read2_method,
     InsertSizeMetrics_adapters_read2__doc__},

    {NULL},
};

static PyType_Slot InsertSizeMetrics_slots[] = {
    {Py_tp_dealloc, (destructor)InsertSizeMetrics_dealloc},
    {Py_tp_new, (newfunc)InsertSizeMetrics__new__},
    {Py_tp_methods, InsertSizeMetrics_methods},
    {Py_tp_members, InsertSizeMetrics_members},
    {0, NULL},
};

static PyType_Spec InsertSizeMetrics_spec = {
    .name = "_qc.InsertSizeMetrics",
    .basicsize = sizeof(InsertSizeMetrics),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT,
    .slots = InsertSizeMetrics_slots,
};

/*************************
 * MODULE INITIALIZATION *
 *************************/

/* A C implementation of from module_name import class_name*/
static PyTypeObject *
ImportClassFromModule(const char *module_name, const char *class_name)
{
    PyObject *module = PyImport_ImportModule(module_name);
    if (module == NULL) {
        return NULL;
    }
    PyObject *type_object = PyObject_GetAttrString(module, class_name);
    Py_DECREF(module);
    if (type_object == NULL) {
        return NULL;
    }
    if (!PyType_CheckExact(type_object)) {
        PyErr_Format(PyExc_RuntimeError, "%s.%s is not a type class but, %R",
                     module_name, class_name, Py_TYPE((PyObject *)type_object));
        return NULL;
    }
    return (PyTypeObject *)type_object;
}

/**
 * @brief Add a new type to the module initiated from spec. Return NULL on
 * failure. Returns the type on success.
 */
static PyTypeObject *
python_module_add_type_spec(PyObject *module, PyType_Spec *spec)
{
    const char *class_name = strchr(spec->name, '.');
    if (class_name == NULL) {
        return NULL;
    }
    class_name += 1;  // Use the part after the dot.

    PyTypeObject *type =
        (PyTypeObject *)PyType_FromModuleAndSpec(module, spec, NULL);
    if (type == NULL) {
        return NULL;
    }

    if (PyModule_AddObjectRef(module, class_name, (PyObject *)type) != 0) {
        Py_DECREF(type);
        return NULL;
    }
    return type;
}

struct AddressAndSpec {
    PyTypeObject **address;
    PyType_Spec *spec;
};

static int
_qc_exec(PyObject *module)
{
    struct QCModuleState *state = PyModule_GetState(module);

    PyTypeObject *PythonArray = ImportClassFromModule("array", "array");
    if (PythonArray == NULL) {
        return -1;
    }
    state->PythonArray_Type = PythonArray;

    struct AddressAndSpec state_address_and_spec[] = {
        {&state->AdapterCounter_Type, &AdapterCounter_spec},
        {&state->BamParser_Type, &BamParser_spec},
        {&state->DedupEstimator_Type, &DedupEstimator_spec},
        {&state->FastqParser_Type, &FastqParser_spec},
        {&state->FastqRecordView_Type, &FastqRecordView_spec},
        {&state->FastqRecordArrayView_Type, &FastqRecordArrayView_spec},
        {&state->InsertSizeMetrics_Type, &InsertSizeMetrics_spec},
        {&state->NanoporeReadInfo_Type, &NanoporeReadInfo_spec},
        {&state->NanoStats_Type, &NanoStats_spec},
        {&state->NanoStatsIterator_Type, &NanoStatsIterator_spec},
        {&state->OverrepresentedSequences_Type, &OverrepresentedSequences_spec},
        {&state->PerTileQuality_Type, &PerTileQuality_spec},
        {&state->QCMetrics_Type, &QCMetrics_spec},
    };

    size_t state_address_entries =
        sizeof(state_address_and_spec) / sizeof(struct AddressAndSpec);

    for (size_t i = 0; i < state_address_entries; i++) {
        struct AddressAndSpec x = state_address_and_spec[i];
        PyTypeObject **address = x.address;
        PyType_Spec *spec = x.spec;
        PyTypeObject *tp = python_module_add_type_spec(module, spec);
        if (tp == NULL) {
            return -1;
        }
        address[0] = tp;
    }

    int ret = 0;
    ret = PyModule_AddIntConstant(module, "NUMBER_OF_NUCS", NUC_TABLE_SIZE);
    if (ret < 0) {
        return -1;
    }
    ret = PyModule_AddIntConstant(module, "NUMBER_OF_PHREDS", PHRED_TABLE_SIZE);
    if (ret < 0) {
        return -1;
    }
    ret = PyModule_AddIntConstant(module, "TABLE_SIZE",
                                  PHRED_TABLE_SIZE * NUC_TABLE_SIZE);
    if (ret < 0) {
        return -1;
    }
    ret = PyModule_AddIntMacro(module, A);
    if (ret < 0) {
        return -1;
    }
    ret = PyModule_AddIntMacro(module, C);
    if (ret < 0) {
        return -1;
    }
    ret = PyModule_AddIntMacro(module, G);
    if (ret < 0) {
        return -1;
    }
    ret = PyModule_AddIntMacro(module, T);
    if (ret < 0) {
        return -1;
    }
    ret = PyModule_AddIntMacro(module, N);
    if (ret < 0) {
        return -1;
    }
    ret = PyModule_AddIntMacro(module, PHRED_MAX);
    if (ret < 0) {
        return -1;
    }
    ret = PyModule_AddIntMacro(module, MAX_SEQUENCE_SIZE);
    if (ret < 0) {
        return -1;
    }
    ret = PyModule_AddIntMacro(module, DEFAULT_MAX_UNIQUE_FRAGMENTS);
    if (ret < 0) {
        return -1;
    }
    ret = PyModule_AddIntMacro(module, DEFAULT_DEDUP_MAX_STORED_FINGERPRINTS);
    if (ret < 0) {
        return -1;
    }
    ret = PyModule_AddIntMacro(module, DEFAULT_FRAGMENT_LENGTH);
    if (ret < 0) {
        return -1;
    }
    ret = PyModule_AddIntMacro(module, DEFAULT_UNIQUE_SAMPLE_EVERY);
    if (ret < 0) {
        return -1;
    }
    ret = PyModule_AddIntMacro(module, DEFAULT_BASES_FROM_START);
    if (ret < 0) {
        return -1;
    }
    ret = PyModule_AddIntMacro(module, DEFAULT_BASES_FROM_END);
    if (ret < 0) {
        return -1;
    }
    ret = PyModule_AddIntMacro(module, DEFAULT_FINGERPRINT_FRONT_SEQUENCE_LENGTH);
    if (ret < 0) {
        return -1;
    }
    ret = PyModule_AddIntMacro(module, DEFAULT_FINGERPRINT_BACK_SEQUENCE_LENGTH);
    if (ret < 0) {
        return -1;
    }
    ret = PyModule_AddIntMacro(module, DEFAULT_FINGERPRINT_FRONT_SEQUENCE_OFFSET);
    if (ret < 0) {
        return -1;
    }
    ret = PyModule_AddIntMacro(module, DEFAULT_FINGERPRINT_BACK_SEQUENCE_OFFSET);
    if (ret < 0) {
        return -1;
    }
    ret = PyModule_AddIntMacro(module, INSERT_SIZE_MAX_ADAPTER_STORE_SIZE);
    if (ret < 0) {
        return -1;
    }

    ret = PyModule_AddIntMacro(module, DEFAULT_END_ANCHOR_LENGTH);
    if (ret < 0) {
        return -1;
    }
    return 0;
}

static int
_qc_traverse(PyObject *mod, visitproc visit, void *arg)
{
    PyTypeObject **mod_state_types = PyModule_GetState(mod);
    if (mod_state_types == NULL) {
        return -1;
    }
    size_t number_of_types =
        sizeof(struct QCModuleState) / sizeof(PyTypeObject *);
    for (size_t i = 0; i < number_of_types; i++) {
        PyTypeObject *tp = mod_state_types[i];
        Py_VISIT(tp);
    }
    return 0;
}

static int
_qc_clear(PyObject *mod)
{
    PyTypeObject **mod_state_types = PyModule_GetState(mod);
    if (mod_state_types == NULL) {
        return -1;
    }
    size_t number_of_types =
        sizeof(struct QCModuleState) / sizeof(PyTypeObject *);
    for (size_t i = 0; i < number_of_types; i++) {
        Py_XDECREF(mod_state_types[i]);
        mod_state_types[i] = NULL;
    }
    return 0;
}

static void
_qc_free(void *mod)
{
    _qc_clear((PyObject *)mod);
}

static PyModuleDef_Slot _qc_module_slots[] = {
    {Py_mod_exec, _qc_exec},
    {0, NULL},
};

static struct PyModuleDef _qc_module = {
    PyModuleDef_HEAD_INIT,  // TODO: Add traverse etc.
    .m_name = "_qc",
    .m_doc = NULL,
    .m_size = sizeof(struct QCModuleState),
    .m_methods = NULL,
    .m_slots = _qc_module_slots,
    .m_traverse = _qc_traverse,
    .m_clear = _qc_clear,
    .m_free = _qc_free,
};

PyMODINIT_FUNC
PyInit__qc(void)
{
    return PyModuleDef_Init(&_qc_module);
}
