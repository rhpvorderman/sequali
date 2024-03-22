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
#include "stdbool.h"

#include "math.h"
#include "score_to_error_rate.h"
#include "murmur3.h"
#include "wanghash.h"

#ifdef __SSE2__
#include "emmintrin.h"
#endif

#ifdef __SSSE3__
#include "tmmintrin.h"
#endif

#if (PY_VERSION_HEX < 0x03090000)
    #define Py_SET_REFCNT(op, count) (Py_REFCNT(op) = count)
    #define Py_SET_SIZE(op, size) (Py_SIZE(op) = size)
    #define Py_SET_TYPE(op, type) (Py_TYPE(op) = type)
#endif

/* Pointers to types that will be imported in the module initialization section */

static PyTypeObject *PythonArray;  // array.array

#define PHRED_MAX 93


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
    for (size_t i=0; i < length; i++) {
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

/**
 * @brief Check if a string of given length only contains ASCII characters.
 *
 * @param string A char pointer to the start of the string.
 * @param length The length of the string. This funtion does not check for 
 *               terminating NULL bytes.
 * @returns 1 if the string is ASCII-only, 0 otherwise.
 */
static int
string_is_ascii(const char * string, size_t length) {
    // By performing bitwise OR on all characters in 8-byte chunks (16-byte 
    // with SSE2) we can
    // determine ASCII status in a non-branching (except the loops) fashion.
    uint64_t all_chars = 0;
    const char *cursor = string;
    const char *string_end_ptr = string + length;
    const char *string_8b_end_ptr = string_end_ptr - sizeof(uint64_t); 
    int non_ascii_in_vec = 0;
    #ifdef __SSE2__
    const char *string_16b_end_ptr = string_end_ptr - sizeof(__m128i);
    __m128i vec_all_chars = _mm_setzero_si128();
    while (cursor < string_16b_end_ptr) {
        __m128i loaded_chars = _mm_loadu_si128((__m128i *)cursor);
        vec_all_chars = _mm_or_si128(loaded_chars, vec_all_chars);
        cursor += sizeof(__m128i);
    }
    non_ascii_in_vec = _mm_movemask_epi8(vec_all_chars);
    #endif

    while (cursor < string_8b_end_ptr) {
        all_chars |= *(uint64_t *)cursor;
        cursor += sizeof(uint64_t);
    }
    while (cursor < string_end_ptr) {
        all_chars |= *cursor;
        cursor += 1;
    }
    return !(non_ascii_in_vec + (all_chars & ASCII_MASK_8BYTE));
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
posix_gm_time(time_t year, time_t month, time_t mday, time_t hour, time_t minute, time_t second)
{
    /* Following code is only true for years equal or greater than 1970*/
    if (year < 1970 || month < 1 || month > 12) {
        return -1;
    } 
    year -= 1900; // Years are relative to 1900
    static const int mday_to_yday[12] = {
        0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};
    time_t yday = mday_to_yday[month - 1] + mday - 1;
    return second + minute*60 + hour*3600 + yday*86400 +
    (year-70)*31536000 + ((year-69)/4)*86400 -
    ((year-1)/100)*86400 + ((year+299)/400)*86400;
}

/**
 * @brief Convert a timestring in Nanopore format to a timestamp. 
 * 
 * @param time_string A string in year-month-dateThour:minute:secondZ format.
 * @return time_t The unix timestamp, -1 on failure. Nanopore was not invented 
 *          on New Year's eve 1969 so this should not lead to confusion ;-).
 */
static time_t time_string_to_timestamp(const uint8_t *time_string) {
    /* Time format used 2019-01-26T18:52:46Z
       Could be parsed with sscanf, but it is much quicker to completely inline
       the call by using an inlinable function. */
    const uint8_t *s = time_string;
    Py_ssize_t year = unsigned_decimal_integer_from_string(s, 4);
    Py_ssize_t month = unsigned_decimal_integer_from_string(s+5, 2);
    Py_ssize_t day = unsigned_decimal_integer_from_string(s+8, 2);
    Py_ssize_t hour = unsigned_decimal_integer_from_string(s+11, 2);
    Py_ssize_t minute = unsigned_decimal_integer_from_string(s+14, 2);
    Py_ssize_t second = unsigned_decimal_integer_from_string(s+17, 2);
    /* If one of year, month etc. is -1 the signed bit is set. Bitwise OR 
       allows checking them all at once for this. */
    if ((year | month | day | hour | minute | second) < 0 || 
         s[4] != '-' || s[7] != '-' || s[10] != 'T' || s[13] != ':' || 
         s[16] != ':') {
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
    switch(tz_part[0]) {
        case 'Z':
            /* UTC No special code needed. */
            break;
        case '+':
        case '-':
            offset_hours = unsigned_decimal_integer_from_string(tz_part + 1, 2);
            offset_minutes = unsigned_decimal_integer_from_string(tz_part + 4, 2);
            if ((offset_hours | offset_minutes) < 0 || tz_part[3] != ':' ) {
                return -1;
            }
            if ((tz_part[0]) == '+') {
                hour += offset_hours;
                minute += offset_minutes;
            } else {
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
    /* Store the accumulated error once calculated so it can be reused by
       the NanoStats module */
    double accumulated_error_rate;
    // Nanopore specific metadata
    time_t start_time;
    float duration;
    int32_t channel;
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
        return NULL;
    }

    double accumulated_error_rate = 0.0;
    for (size_t i=0; i < sequence_length; i++) {
        uint8_t q = qualities[i] - 33;
        if (q > PHRED_MAX) {
                PyErr_Format(
                    PyExc_ValueError, 
                    "Not a valid phred character: %c", qualities[i]
                );
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

    uint8_t *buffer = (uint8_t *)PyBytes_AS_STRING(bytes_obj);
    self->meta.record_start = buffer;
    self->meta.name_length = name_length;
    self->meta.sequence_offset = 2 + name_length;
    self->meta.sequence_length = sequence_length;
    self->meta.qualities_offset = 5 + name_length + sequence_length;
    self->meta.accumulated_error_rate = accumulated_error_rate;
    self->meta.duration = 0.0;
    self->meta.start_time = 0;
    self->meta.channel = -1;
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

PyDoc_STRVAR(FastqRecordView_name__doc__,
"name($self)\n"
"--\n"
"\n"
"Returns the FASTQ header.\n"
);

static PyObject *
FastqRecordView_name(FastqRecordView *self, PyObject *Py_UNUSED(ignore))
{
    PyObject *result = PyUnicode_New(self->meta.name_length, 127);
    if (result == NULL) {
        return NULL;
    }
    memcpy(PyUnicode_DATA(result), 
           self->meta.record_start + 1, 
           self->meta.name_length);
    return result;
}

PyDoc_STRVAR(FastqRecordView_sequence__doc__,
"sequence($self)\n"
"--\n"
"\n"
"Returns the FASTQ nucleotide sequence.\n"
);

static PyObject *
FastqRecordView_sequence(FastqRecordView *self, PyObject *Py_UNUSED(ignore))
{
    PyObject *result = PyUnicode_New(self->meta.sequence_length, 127);
    if (result == NULL) {
        return NULL;
    }
    memcpy(PyUnicode_DATA(result), 
           self->meta.record_start + 
           self->meta.sequence_offset, 
           self->meta.sequence_length);
    return result;
}

PyDoc_STRVAR(FastqRecordView_qualities__doc__,
"qualities($self)\n"
"--\n"
"\n"
"Returns the FASTQ phred encoded qualities as a string.\n"
);

static PyObject *
FastqRecordView_qualities(FastqRecordView *self, PyObject *Py_UNUSED(ignore))
{
    PyObject *result = PyUnicode_New(self->meta.sequence_length, 127);
    if (result == NULL) {
        return NULL;
    }
    memcpy(PyUnicode_DATA(result), 
           self->meta.record_start + self->meta.qualities_offset, 
           self->meta.sequence_length);
    return result;
}

static PyMethodDef FastqRecordView_methods[] = {
    {"name", (PyCFunction)FastqRecordView_name, METH_NOARGS,
     FastqRecordView_name__doc__},
    {"sequence", (PyCFunction)FastqRecordView_sequence, METH_NOARGS,
     FastqRecordView_sequence__doc__},
    {"qualities", (PyCFunction)FastqRecordView_qualities, METH_NOARGS, 
     FastqRecordView_qualities__doc__},
    {NULL}
};

static PyMemberDef FastqRecordView_members[] = {
    {"obj", T_OBJECT, offsetof(FastqRecordView, obj), READONLY,
     "The underlying buffer where the fastq record is located"},
    {NULL},
};


static PyTypeObject FastqRecordView_Type = {
    .tp_name = "_qc.FastqRecordView",
    .tp_basicsize = sizeof(FastqRecordView),
    .tp_dealloc = (destructor)FastqRecordView_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = (newfunc)FastqRecordView__new__,
    .tp_methods = FastqRecordView_methods,
    .tp_members = FastqRecordView_members,
};

static inline int 
FastqRecordView_CheckExact(void *obj) 
{
    return Py_TYPE(obj) == &FastqRecordView_Type;
} 

static PyObject *
FastqRecordView_FromFastqMetaAndObject(struct FastqMeta *meta, PyObject *object)
{
    FastqRecordView *self = PyObject_New(FastqRecordView, &FastqRecordView_Type);
    if (self == NULL) {
        return PyErr_NoMemory();
    }
    memcpy(&self->meta, meta, sizeof(struct FastqMeta));
    Py_XINCREF(object);
    self->obj = object; 
    return (PyObject *)self;
}

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
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyTypeObject FastqRecordArrayView_Type;

static PyObject *
FastqRecordArrayView_FromPointerSizeAndObject(
    struct FastqMeta *records, size_t number_of_records, PyObject *obj) 
{
    size_t size = number_of_records * sizeof(struct FastqMeta);
    FastqRecordArrayView *self = PyObject_Malloc(sizeof(FastqRecordArrayView) + size);
    if (self == NULL) {
        return PyErr_NoMemory();
    }
    Py_SET_REFCNT(self, 1);
    Py_SET_TYPE(self, &FastqRecordArrayView_Type);
    Py_SET_SIZE(self, number_of_records);
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
    PyObject *view_fastseq = PySequence_Fast(view_items_obj, 
        "view_items should be iterable");
    if (view_fastseq == NULL) {
        return NULL;
    }
    Py_ssize_t number_of_items = PySequence_Fast_GET_SIZE(view_fastseq);
    PyObject **items = PySequence_Fast_ITEMS(view_fastseq);
    size_t total_memory_size = 0;
    for (Py_ssize_t i = 0; i < number_of_items; i++) {
        PyObject *item = items[i];
        if (Py_TYPE(item) != &FastqRecordView_Type) {
            PyErr_Format(
                PyExc_TypeError, 
                "Expected an iterable of FastqRecordView objects, but item %z "
                "is of type %s: %R", i, Py_TYPE(item)->tp_name, item);
            return NULL;
        }
        FastqRecordView *record = (FastqRecordView *) item;
        size_t memory_size = 6 + record->meta.name_length + 
            record->meta.sequence_length * 2;
        total_memory_size += memory_size;
    }
    PyObject *obj = PyBytes_FromStringAndSize(NULL, total_memory_size);
    if (obj == NULL) {
        return PyErr_NoMemory();
    }
    FastqRecordArrayView *record_array = 
        (FastqRecordArrayView *)
        FastqRecordArrayView_FromPointerSizeAndObject(NULL, number_of_items, obj);
    if (record_array == NULL) {
        Py_DECREF(obj); 
        return NULL;
    }
    char *record_ptr = PyBytes_AS_STRING(obj);
    struct FastqMeta *metas = record_array->records;
    for (Py_ssize_t i=0; i < number_of_items; i++) {
        FastqRecordView *record = (FastqRecordView *)items[i];
        struct FastqMeta meta = record->meta;
        record_ptr[0] = '@';
        record_ptr += 1; 
        memcpy(record_ptr, meta.record_start + 1, meta.name_length);
        record_ptr += meta.name_length;
        record_ptr[0] = '\n';
        record_ptr += 1;
        memcpy(record_ptr, meta.record_start + meta.sequence_offset, meta.sequence_length);
        record_ptr += meta.sequence_length;
        record_ptr[0] = '\n';
        record_ptr[1] = '+';
        record_ptr[2] = '\n';
        record_ptr += 3;
        memcpy(record_ptr, meta.record_start + meta.qualities_offset, meta.sequence_length);
        record_ptr += meta.sequence_length;
        record_ptr[0] = '\n';
        record_ptr += 1;
        memcpy(metas + i, &record->meta, sizeof(struct FastqMeta));
    }
    return (PyObject *)record_array;
}

static PyObject *
FastqRecordArrayView__get_item__(FastqRecordArrayView *self, Py_ssize_t i)
{
    Py_ssize_t size = Py_SIZE(self);
    if (i < 0) {
        i = size + i;
    }
    if (i < 0 || i >= size) {
        PyErr_SetString(PyExc_IndexError, "array index out of range");
        return NULL;
    }
    return FastqRecordView_FromFastqMetaAndObject(self->records + i, self->obj);
}

static inline Py_ssize_t FastqRecordArrayView__length__(
        FastqRecordArrayView *self) {
    return Py_SIZE(self);
}

static inline int 
FastqRecordArrayView_CheckExact(void *obj) {
    return Py_TYPE(obj) == &FastqRecordArrayView_Type;
}

static PySequenceMethods FastqRecordArrayView_sequence_methods = {
    .sq_item = (ssizeargfunc)FastqRecordArrayView__get_item__,
    .sq_length = (lenfunc)FastqRecordArrayView__length__,
};

static PyMemberDef FastqRecordArrayView_members[] = {
    {"obj", T_OBJECT, offsetof(FastqRecordArrayView, obj), READONLY,
     "The underlying buffer where the fastq records are located"},
    {NULL},
};

static PyTypeObject FastqRecordArrayView_Type = {
    .tp_name = "_qc.FastqRecordArrayView",
    .tp_dealloc = (destructor)FastqRecordArrayView_dealloc,
    .tp_basicsize = sizeof(FastqRecordArrayView),
    .tp_itemsize = sizeof(struct FastqMeta),
    .tp_as_sequence = &FastqRecordArrayView_sequence_methods,
    .tp_new = FastqRecordArrayView__new__,
    .tp_members = FastqRecordArrayView_members,
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
    Py_TYPE(self)->tp_free((PyObject *)self);
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
    self->record_start = (uint8_t *)PyBytes_AS_STRING(buffer_obj);
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

static PyObject *
FastqParser__next__(FastqParser *self) 
{
    uint8_t *record_start = self->record_start;
    uint8_t *buffer_end = self->buffer_end;
    size_t parsed_records = 0;
    PyObject *new_buffer_obj = NULL;
    while (parsed_records == 0) {
        size_t leftover_size = buffer_end - record_start;
        size_t read_in_size;
        if (leftover_size >= self->read_in_size) {
        	// A FASTQ record does not fit, enlarge the buffer
            read_in_size = self->read_in_size;
        } else {
        	// Fill up the buffer up to read_in_size
        	read_in_size = self->read_in_size - leftover_size;
        }
        Py_ssize_t new_buffer_size = leftover_size + read_in_size;
        if (new_buffer_obj == NULL) {
            new_buffer_obj = PyBytes_FromStringAndSize(NULL, new_buffer_size);
            if (new_buffer_obj == NULL) {
                return NULL;
            }
            memcpy(PyBytes_AS_STRING(new_buffer_obj), record_start, leftover_size);
        } else {
            if (_PyBytes_Resize(&new_buffer_obj, new_buffer_size) == -1) {
                return NULL;
            }
        }
        uint8_t *new_buffer = (uint8_t *)PyBytes_AS_STRING(new_buffer_obj);

        PyObject *remaining_space_view = PyMemoryView_FromMemory(
            (char *)new_buffer + leftover_size, read_in_size, PyBUF_WRITE);
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
        Py_ssize_t actual_buffer_size = leftover_size + read_bytes;
        if (actual_buffer_size < new_buffer_size) {
            if (_PyBytes_Resize(&new_buffer_obj, actual_buffer_size) == -1) {
                return NULL;
            }
        }
        new_buffer = (uint8_t *)PyBytes_AS_STRING(new_buffer_obj);
        new_buffer_size = actual_buffer_size;
        if (!string_is_ascii((char *)new_buffer + leftover_size, read_bytes)) {
            Py_ssize_t pos;
            for (pos=leftover_size; pos<new_buffer_size; pos+=1) {
                if (new_buffer[pos] & ASCII_MASK_1BYTE) {
                    break;
                }
            }
            PyErr_Format(
                PyExc_ValueError, 
                "Found non-ASCII character in file: %c", new_buffer[pos]
            );
            Py_DECREF(new_buffer_obj);
            return NULL;
        }

        if (new_buffer_size == 0) {
            // Entire file is read
            PyErr_SetNone(PyExc_StopIteration);
            Py_DECREF(new_buffer_obj);
            return NULL;
        } else if (read_bytes == 0) {
            // Incomplete record at the end of file;
            PyErr_Format(
                PyExc_EOFError,
                "Incomplete record at the end of file %s", 
                new_buffer);
            Py_DECREF(new_buffer_obj);
            return NULL;
        }
        record_start = new_buffer;
        buffer_end = record_start + new_buffer_size;

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
                Py_DECREF(new_buffer_obj);
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
                Py_DECREF(new_buffer_obj);
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
                PyObject *record_name_obj = PyUnicode_DecodeASCII((char *)record_start + 1, name_length, NULL);
                PyErr_Format(
                    PyExc_ValueError,
                    "Record sequence and qualities do not have equal length, %R",
                    record_name_obj
                );
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
            meta->record_start = record_start;
            meta->name_length = name_length;
            meta->sequence_offset = sequence_start - record_start;
            meta->sequence_length = sequence_length;
            meta->qualities_offset = qualities_start - record_start;
            meta->accumulated_error_rate = 0.0;
            meta->channel=-1;
            meta->duration=0.0;
            meta->start_time=0;
            record_start = qualities_end + 1;
        }
    }
    PyObject *tmp = self->buffer_obj;
    self->buffer_obj = new_buffer_obj;
    Py_DECREF(tmp);
    self->record_start = record_start;
    self->buffer_end = buffer_end;
    return FastqRecordArrayView_FromPointerSizeAndObject(
        self->meta_buffer, parsed_records, new_buffer_obj);
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
 * BAM PARSER *
 * ************/

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
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
BamParser__new__(PyTypeObject *type, PyObject *args, PyObject *kwargs) 
{
    PyObject *file_obj = NULL;
    size_t read_in_size = 48 * 1024; // Slightly smaller than BGZF block size
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
    PyObject *magic_and_header_size = PyObject_CallMethod(file_obj, "read", "n", 8);
    if (magic_and_header_size == NULL) {
        return NULL;
    }
    if (!PyBytes_CheckExact(magic_and_header_size)) {
        PyErr_Format(
            PyExc_TypeError,
            "file_obj %R is not a binary IO type, got %s",
            file_obj, Py_TYPE(file_obj)->tp_name
        );
        Py_DECREF(magic_and_header_size);
        return NULL;
    }
    if (PyBytes_GET_SIZE(magic_and_header_size) < 8) {
        PyErr_SetString(PyExc_EOFError, "Truncated BAM file");
        Py_DECREF(magic_and_header_size);
        return NULL;
    }
    uint8_t *file_start = (uint8_t *)PyBytes_AS_STRING(magic_and_header_size);
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
    if (PyBytes_GET_SIZE(header) != l_text) {
        PyErr_SetString(PyExc_EOFError, "Truncated BAM file");
        Py_DECREF(header);
        return NULL;
    }
    PyObject *n_ref_obj = PyObject_CallMethod(file_obj, "read", "n", 4);
    if (PyBytes_GET_SIZE(n_ref_obj) != 4) {
        PyErr_SetString(PyExc_EOFError, "Truncated BAM file");
        Py_DECREF(n_ref_obj);
        Py_DECREF(header);
        return NULL;
    }
    uint32_t n_ref = *(uint32_t *)PyBytes_AS_STRING(n_ref_obj);
    Py_DECREF(n_ref_obj);

    for (size_t i=0; i < n_ref; i++) {
        PyObject *l_name_obj = PyObject_CallMethod(file_obj, "read", "n", 4);
        if (PyBytes_GET_SIZE(l_name_obj) != 4) {
            PyErr_SetString(PyExc_EOFError, "Truncated BAM file");
            Py_DECREF(header);
            return NULL;
        }
        size_t l_name = *(uint32_t *)PyBytes_AS_STRING(l_name_obj);
        Py_DECREF(l_name_obj);
        Py_ssize_t reference_chunk_size = l_name + 4;  // Includes name and uint32_t for size.
        PyObject *reference_chunk = PyObject_CallMethod(file_obj, "read", "n", reference_chunk_size);
        Py_ssize_t actual_reference_chunk_size = PyBytes_GET_SIZE(reference_chunk);
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
BamParser__iter__(BamParser *self) {
    Py_INCREF(self);
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

static void 
decode_bam_sequence(uint8_t *dest, const uint8_t *encoded_sequence, size_t length) 
{
    /* Reuse a trick from sam_internal.h in htslib. Have a table to lookup 
       two characters simultaneously.*/
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
    const uint8_t *dest_end_ptr = dest + length;
    uint8_t *dest_cursor = dest;
    const uint8_t *encoded_cursor = encoded_sequence;
    #ifdef __SSSE3__
    const uint8_t *dest_vec_end_ptr = dest_end_ptr - (2 * sizeof(__m128i));
    __m128i first_upper_shuffle = _mm_setr_epi8(
        0, 0xff, 1, 0xff, 2, 0xff, 3, 0xff, 4, 0xff, 5, 0xff, 6, 0xff, 7, 0xff);
    __m128i first_lower_shuffle = _mm_setr_epi8(
        0xff, 0, 0xff, 1, 0xff, 2, 0xff, 3, 0xff, 4, 0xff, 5, 0xff, 6, 0xff, 7);
    __m128i second_upper_shuffle = _mm_setr_epi8(
        8, 0xff, 9, 0xff, 10, 0xff, 11, 0xff, 12, 0xff, 13, 0xff, 14, 0xff, 15, 0xff);
    __m128i second_lower_shuffle = _mm_setr_epi8(
        0xff, 8, 0xff, 9, 0xff, 10, 0xff, 11, 0xff, 12, 0xff, 13, 0xff, 14, 0xff, 15);
    __m128i nuc_lookup_vec = _mm_lddqu_si128((__m128i *)nuc_lookup);
    /* Work on 16 encoded characters at the time resulting in 32 decoded characters 
       Examples are given for 8 encoded characters A until H to keep it readable.
        Encoded stored as |AB|CD|EF|GH|
        Shuffle into |AB|00|CD|00|EF|00|GH|00| and 
                     |00|AB|00|CD|00|EF|00|GH| 
        Shift upper to the right resulting into
                     |0A|B0|0C|D0|0E|F0|0G|H0| and 
                     |00|AB|00|CD|00|EF|00|GH|
        Merge with or resulting into (X stands for garbage)
                     |0A|XB|0C|XD|0E|XF|0G|XH|
        Bitwise and with 0b1111 leads to:
                     |0A|0B|0C|0D|0E|0F|0G|0H|
        We can use the resulting 4-bit integers as indexes for the shuffle of 
        the nucleotide lookup. */
    while (dest_cursor < dest_vec_end_ptr) {
        __m128i encoded = _mm_lddqu_si128((__m128i *)encoded_cursor);

        __m128i first_upper = _mm_shuffle_epi8(encoded, first_upper_shuffle);
        __m128i first_lower = _mm_shuffle_epi8(encoded, first_lower_shuffle);
        __m128i shifted_first_upper = _mm_srli_epi64(first_upper, 4);
        __m128i first_merged = _mm_or_si128(shifted_first_upper, first_lower);
        __m128i first_indexes = _mm_and_si128(first_merged, _mm_set1_epi8(0b1111));
        __m128i first_nucleotides = _mm_shuffle_epi8(nuc_lookup_vec, first_indexes);
        _mm_storeu_si128((__m128i *)dest_cursor, first_nucleotides);

        __m128i second_upper = _mm_shuffle_epi8(encoded, second_upper_shuffle);
        __m128i second_lower = _mm_shuffle_epi8(encoded, second_lower_shuffle);
        __m128i shifted_second_upper = _mm_srli_epi64(second_upper, 4);
        __m128i second_merged = _mm_or_si128(shifted_second_upper, second_lower);
        __m128i second_indexes = _mm_and_si128(second_merged, _mm_set1_epi8(0b1111));
        __m128i second_nucleotides = _mm_shuffle_epi8(nuc_lookup_vec, second_indexes);
        _mm_storeu_si128((__m128i *)(dest_cursor + 16), second_nucleotides);

        encoded_cursor += sizeof(__m128i);
        dest_cursor += 2 * sizeof(__m128i);
    }
    #endif
    /* Do two at the time until it gets to the last even address. */
    const uint8_t *dest_end_ptr_twoatatime = dest + (length & (~1ULL));
    while (dest_cursor < dest_end_ptr_twoatatime) {
        /* According to htslib, size_t cast helps the optimizer. 
           Code confirmed to indeed run faster. */
        memcpy(dest_cursor, code2base + ((size_t)*encoded_cursor * 2), 2);
        dest_cursor += 2;
        encoded_cursor += 1;
    }
    assert((dest_end_ptr - dest_cursor) < 2);
    if (dest_cursor != dest_end_ptr) {
        /* There is a single encoded nuc left */
        uint8_t encoded_nucs = *encoded_cursor;
        uint8_t upper_nuc_index = encoded_nucs >> 4;
        dest_cursor[0] = nuc_lookup[upper_nuc_index];
    }
}

static void 
decode_bam_qualities(uint8_t *dest, const uint8_t *encoded_qualities, size_t length) 
{
    const uint8_t *end_ptr = encoded_qualities + length;
    const uint8_t *cursor = encoded_qualities;
    uint8_t *dest_cursor = dest; 
    #ifdef __SSE2__
    const uint8_t *vec_end_ptr = end_ptr - sizeof(__m128i);
    while (cursor < vec_end_ptr) {
        __m128i quals = _mm_loadu_si128((__m128i *)cursor);
        __m128i phreds = _mm_add_epi8(quals, _mm_set1_epi8(33));
        _mm_storeu_si128((__m128i *)dest_cursor, phreds);
        cursor += sizeof(__m128i);
        dest_cursor += sizeof(__m128i);
    }
    #endif
    while (cursor < end_ptr) {
        *dest_cursor = *cursor + 33; 
        cursor += 1;
        dest_cursor += 1;    
    }
}

static int
bam_tags_to_fastq_meta(const uint8_t *tags, size_t tags_length, struct FastqMeta *meta)
{
    meta->channel = -1;
    meta->duration = 0.0;
    meta->start_time = 0;
    while (tags_length > 0) {
        if (tags_length < 4) {
            PyErr_SetString(PyExc_ValueError, "truncated tags");
            return -1;
        }
        const uint8_t *tag_id = tags;
        uint8_t tag_type = tags[2];
        bool is_array = false;
        const uint8_t *value_start = tags + 3;
        uint32_t array_length = 1;
        if (tag_type == 'B') {
            is_array = true; 
            value_start = tags + 8;
            tag_type = tags[3];
            if (tags_length < 8) {
                PyErr_SetString(PyExc_ValueError, "truncated tags");
                return -1;
            }
            array_length = *(uint32_t *)(tags + 4);
        };
        size_t value_length;
        switch (tag_type) {
            case 'A':
                value_length = 1;
                break;
            case 'c':
            case 'C':
                /* A very annoying habit of htslib to store a tag in the 
                   smallest possible size rather than being consistent. */
                value_length = 1;
                if (memcmp(tag_id, "ch", 2) == 0 && tags_length >= 4) {
                    meta->channel = *(uint8_t *)(value_start); 
                }
                break; 
            case 's':
            case 'S': 
                value_length = 2;
                if (memcmp(tag_id, "ch", 2) == 0 && tags_length >= 5) {
                    meta->channel = *(uint16_t *)(value_start); 
                }
                break;
            case 'I':
            case 'i':
                if (memcmp(tag_id, "ch", 2) == 0 && tags_length >= 7) {
                    meta->channel = *(uint32_t *)(value_start); 
                }   
                value_length = 4;
                break;
            case 'f':
                if (memcmp(tag_id, "du", 2) == 0 && tags_length >= 7) {
                    meta->duration = *(float *)(value_start); 
                }
                value_length = 4;
                break;
            case 'Z':
            case 'H':
                if (is_array) {
                    PyErr_Format(PyExc_ValueError, "Invalid type for array %c", tag_type);
                    return -1;
                }
                uint8_t *string_end = memchr(value_start, 0, tags_length - 3);
                if (string_end == NULL) {
                    PyErr_SetString(PyExc_ValueError, "truncated tags");
                    return -1;
                }
                value_length = (string_end - value_start) + 1; // +1 for terminating null
                if (memcmp(tag_id, "st", 2) == 0) {
                    meta->start_time = time_string_to_timestamp(value_start);    
                }
                break;
            default:
                PyErr_Format(PyExc_ValueError, "Unknown tag type %c", tag_type);
                return -1;
        }
        size_t this_tag_length = (value_start - tags) + array_length * value_length;
        if (this_tag_length > tags_length) {
            PyErr_SetString(PyExc_ValueError, "truncated tags");
            return -1;        
        }
        tags = tags + this_tag_length;
        tags_length -= this_tag_length;
    }
    return 0;
}

static PyObject *
BamParser__next__(BamParser *self) {
    uint8_t *record_start = self->record_start;
    uint8_t *buffer_end = self->buffer_end;
    size_t leftover_size = buffer_end - record_start;
    memmove(self->read_in_buffer, record_start, leftover_size);
    record_start = self->read_in_buffer;
    buffer_end = record_start + leftover_size;
    size_t parsed_records = 0;
    PyObject *fastq_buffer_obj = NULL;

    while (parsed_records == 0) {
        /* Keep expanding input buffer until at least one record is parsed */
        size_t read_in_size;
        leftover_size = buffer_end - record_start;
        if (leftover_size >= 4) {
            // Immediately check how much the block is to load enough data;
            uint32_t block_size = *(uint32_t *)record_start;
            read_in_size = Py_MAX(block_size, self->read_in_size);
        } else {
        	// Fill up the buffer up to read_in_size
        	read_in_size = self->read_in_size - leftover_size;
        }
        size_t minimum_space_required = leftover_size + read_in_size;
        if (minimum_space_required > self->read_in_buffer_size) {
            uint8_t *tmp_read_in_buffer = PyMem_Realloc(self->read_in_buffer, minimum_space_required);
            if (tmp_read_in_buffer == NULL) {
                Py_XDECREF(fastq_buffer_obj);
                return PyErr_NoMemory();
            }
            self->read_in_buffer = tmp_read_in_buffer;
            self->read_in_buffer_size = minimum_space_required;
        }
        PyObject *buffer_view = PyMemoryView_FromMemory((char *)self->read_in_buffer + leftover_size, read_in_size, PyBUF_WRITE);
        if (buffer_view == NULL) {
            return NULL;
        }
        PyObject *read_bytes_obj = PyObject_CallMethod(self->file_obj, "readinto", "O", buffer_view);
        Py_DECREF(buffer_view);
        if (read_bytes_obj == NULL) {
            Py_XDECREF(fastq_buffer_obj);
            return NULL;
        }
        Py_ssize_t read_bytes = PyLong_AsSsize_t(read_bytes_obj);
        Py_DECREF(read_bytes_obj);
        size_t new_buffer_size = leftover_size + read_bytes;
        if (new_buffer_size == 0) {
            // Entire file is read
            PyErr_SetNone(PyExc_StopIteration);
            Py_XDECREF(fastq_buffer_obj);
            return NULL;
        } else if (read_bytes == 0) {
            PyObject *remaining_obj = PyBytes_FromStringAndSize((char *)self->read_in_buffer, leftover_size);
            PyErr_Format(
                PyExc_EOFError,
                "Incomplete record at the end of file %R", 
                remaining_obj);
            Py_DECREF(remaining_obj);
            Py_XDECREF(fastq_buffer_obj);
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
        Py_ssize_t fastq_buffer_size = (new_buffer_size * 4 + 2) / 3;
        if (fastq_buffer_obj == NULL) {
            fastq_buffer_obj = PyBytes_FromStringAndSize(NULL, fastq_buffer_size);
            if (fastq_buffer_obj == NULL) {
                Py_XDECREF(fastq_buffer_obj);
                return PyErr_NoMemory();
            }
        } else {
            if (_PyBytes_Resize(&fastq_buffer_obj, fastq_buffer_size) < 0) {
                Py_XDECREF(fastq_buffer_obj);
                return NULL;
            }
        }
        uint8_t *fastq_buffer_record_start = (uint8_t *)PyBytes_AS_STRING(fastq_buffer_obj);

        while (1) {
            if (record_start + 4 >= buffer_end) {
                break;  // Not enough bytes to read block_size
            }
            struct BamRecordHeader *header = (struct BamRecordHeader *)record_start;
            uint8_t *record_end = record_start + 4 + header->block_size;
            if (record_end > buffer_end) {
                break;
            }
            uint8_t *bam_name_start = record_start + sizeof(struct BamRecordHeader);
            uint32_t name_length = header->l_read_name;
            uint8_t *bam_seq_start = bam_name_start + name_length + 
                                     header->n_cigar_op * sizeof(uint32_t);
            uint32_t seq_length = header->l_seq;
            uint32_t encoded_seq_length = (seq_length + 1) / 2;
            uint8_t *bam_qual_start = bam_seq_start + encoded_seq_length;
            fastq_buffer_record_start[0] = '@';
            uint8_t *fastq_buffer_cursor = fastq_buffer_record_start + 1;
            uint8_t *tag_start = bam_qual_start + seq_length;
            size_t tags_length = record_end - tag_start;
            if (name_length > 0) {
                name_length -= 1;  /* Includes terminating NULL byte */
                memcpy(fastq_buffer_cursor, bam_name_start, name_length);
            }
            /* The + and newlines are unncessary, but also do not require much 
               space and compute time. So keep the in-memory presentation as 
               a FASTQ record for homogeneity with FASTQ input. */
            fastq_buffer_cursor += name_length;
            fastq_buffer_cursor[0] = '\n';
            fastq_buffer_cursor += 1;
            decode_bam_sequence(fastq_buffer_cursor, bam_seq_start, seq_length);
            fastq_buffer_cursor += seq_length;
            memcpy(fastq_buffer_cursor, "\n+\n", 3);
            fastq_buffer_cursor += 3;
            decode_bam_qualities(fastq_buffer_cursor, bam_qual_start, seq_length);
            fastq_buffer_cursor += seq_length;
            fastq_buffer_cursor[0] = '\n';
            fastq_buffer_cursor += 1;

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
            uint32_t sequence_offset = 1 + name_length + 1; // For '@' and '\n'
            uint32_t qualities_offset = sequence_offset + seq_length + 3; // for '\n+\n'
            meta->record_start = fastq_buffer_record_start;
            meta->name_length = name_length;
            meta->sequence_offset = sequence_offset;
            meta->sequence_length = seq_length;
            meta->qualities_offset = qualities_offset;
            meta->accumulated_error_rate = 0.0;
            if (bam_tags_to_fastq_meta(tag_start, tags_length, meta) < 0) {
                return NULL;
            }
            record_start = record_end;
            fastq_buffer_record_start = fastq_buffer_cursor;
        }
    }
    self->record_start = record_start;
    self->buffer_end = buffer_end;
    PyObject *record_array = FastqRecordArrayView_FromPointerSizeAndObject(
        self->meta_buffer, parsed_records, fastq_buffer_obj);
    Py_DECREF(fastq_buffer_obj);
    return record_array;
}

static PyMemberDef BamParser_members[] = {
    {"header", T_OBJECT_EX, offsetof(BamParser, header), READONLY, 
     "The BAM header"},
    {NULL}
};

PyTypeObject BamParser_Type = {
    .tp_name = "_qc.BamParser",
    .tp_basicsize = sizeof(BamParser),
    .tp_dealloc = (destructor)BamParser_dealloc,
    .tp_new = BamParser__new__,
    .tp_iter = (iternextfunc)BamParser__iter__,
    .tp_iternext = (iternextfunc)BamParser__next__,
    .tp_members = BamParser_members,
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
#define PHRED_LIMIT 47
#define PHRED_TABLE_SIZE ((PHRED_LIMIT / 4) + 1)

typedef uint16_t staging_base_table[NUC_TABLE_SIZE];
typedef uint16_t staging_phred_table[PHRED_TABLE_SIZE];
typedef uint64_t base_table[NUC_TABLE_SIZE];
typedef uint64_t phred_table[PHRED_TABLE_SIZE];

static inline uint8_t phred_to_index(uint8_t phred) {
    if (phred > PHRED_LIMIT){
        phred = PHRED_LIMIT;
    }
    return phred >> 2;
}

typedef struct _QCMetricsStruct {
    PyObject_HEAD
    uint8_t phred_offset;
    uint16_t staging_count;
    size_t max_length;
    staging_base_table *staging_base_counts;
    staging_phred_table *staging_phred_counts;
    base_table *base_counts;
    phred_table *phred_counts;
    size_t number_of_reads;
    uint64_t gc_content[101];
    uint64_t phred_scores[PHRED_MAX + 1];
} QCMetrics;

static void
QCMetrics_dealloc(QCMetrics *self) {
    PyMem_Free(self->staging_base_counts);
    PyMem_Free(self->staging_phred_counts);
    PyMem_Free(self->base_counts);
    PyMem_Free(self->phred_counts);
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
    self->staging_base_counts = NULL;
    self->staging_phred_counts = NULL;
    self->base_counts = NULL;
    self->phred_counts = NULL;
    self->number_of_reads = 0;
    memset(self->gc_content, 0, 101 * sizeof(uint64_t));
    memset(self->phred_scores, 0, (PHRED_MAX + 1) * sizeof(uint64_t));
    return (PyObject *)self;
}

static int
QCMetrics_resize(QCMetrics *self, Py_ssize_t new_size) 
{
    staging_base_table *staging_base_tmp = PyMem_Realloc(
        self->staging_base_counts, new_size *sizeof(staging_base_table));
    staging_phred_table *staging_phred_tmp = PyMem_Realloc(
        self->staging_phred_counts, new_size * sizeof(staging_phred_table));
    base_table *base_table_tmp = PyMem_Realloc(
        self->base_counts, new_size * sizeof(base_table));
    phred_table *phred_table_tmp = PyMem_Realloc( 
        self->phred_counts, new_size *sizeof(phred_table));

    if (staging_base_tmp == NULL || staging_phred_tmp == NULL || base_table_tmp == NULL || phred_table_tmp == NULL) {
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
    memset(staging_phred_tmp + old_size, 0, new_slots * sizeof(staging_phred_table));
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
QCMetrics_flush_staging(QCMetrics *self) {
    if (self->staging_count == 0) {
        return;
    }
    uint64_t *base_counts = (uint64_t *)self->base_counts;
    uint16_t *staging_base_counts = (uint16_t *)self->staging_base_counts;
    size_t number_of_base_slots = self->max_length * NUC_TABLE_SIZE;
    for (size_t i=0; i < number_of_base_slots; i++) {
        base_counts[i] += staging_base_counts[i];
    }
    memset(staging_base_counts, 0, number_of_base_slots * sizeof(uint16_t));

    uint64_t *phred_counts = (uint64_t *)self->phred_counts;
    uint16_t *staging_phred_counts = (uint16_t *)self->staging_phred_counts;
    size_t number_of_phred_slots = self->max_length * PHRED_TABLE_SIZE;
    for (size_t i=0; i < number_of_phred_slots; i++) {
        phred_counts[i] += staging_phred_counts[i];
    }
    memset(staging_phred_counts, 0, number_of_phred_slots * sizeof(uint16_t));

    self->staging_count = 0;
}

#ifdef __SSE2__
static inline size_t horizontal_add_epu8(__m128i vec) {
    /* _mm_sad_epu8 calculates absolute differences between a and b and then 
       summes the lower 8 bytes horizontally and saves it is in the lower 16 
       bits of the lower 64-bit integer. It does the same for the upper 8 
       bytes and stores it in the upper 64-bit integer. 
       _mm_cvtsi128_si64 gets the lower 64-bit integer. Using 
       _mm_bsrli_si128 we can shift the result 8 bytes to the right, resulting
       in the upper integer being in the place of the lower integer. */
    __m128i hadd = _mm_sad_epu8(vec, _mm_setzero_si128());
    uint64_t lower_count = _mm_cvtsi128_si64(hadd);
    uint64_t upper_count = _mm_cvtsi128_si64(_mm_bsrli_si128(hadd, 8));
    return lower_count + upper_count;
}
#endif

static inline int 
QCMetrics_add_meta(QCMetrics *self, struct FastqMeta *meta)
{
    const uint8_t *record_start = meta->record_start;
    size_t sequence_length = meta->sequence_length;
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
    const uint8_t *sequence_end_ptr = sequence + sequence_length;
    // uint32_t is ample as the maximum length of a sequence is saved in a uint32_r
    uint32_t base_counts[NUC_TABLE_SIZE] = {0, 0, 0, 0, 0};
    #ifdef __SSE2__
    const uint8_t *sequence_vec_end_ptr = sequence_end_ptr - sizeof(__m128i);
    while (sequence_ptr < sequence_vec_end_ptr) {
        /* Store nucleotide in count vectors. This means we can do at most 
           255 loop as the 8-bit integers can become saturated. After 255 loops
           the result is flushed. */
        size_t remaining_length = sequence_vec_end_ptr - sequence_ptr;
        size_t remaining_vecs = (remaining_length + 15) / sizeof(__m128i);
        size_t iterations = Py_MIN(255, remaining_vecs);
        register __m128i a_counts = _mm_setzero_si128();
        register __m128i c_counts = _mm_setzero_si128();
        register __m128i g_counts = _mm_setzero_si128();
        register __m128i t_counts = _mm_setzero_si128();
        register __m128i all1 = _mm_set1_epi8(1);
        for (size_t i=0; i<iterations; i++) {
            __m128i nucleotides = _mm_loadu_si128((__m128i *)sequence_ptr);
            // This will make all the nucleotides uppercase.
            nucleotides = _mm_and_si128(nucleotides, _mm_set1_epi8(223)); 
            __m128i a_nucs = _mm_cmpeq_epi8(nucleotides, _mm_set1_epi8('A'));
            __m128i a_positions = _mm_and_si128(a_nucs, all1);
            a_counts = _mm_add_epi8(a_counts, a_positions);
            __m128i c_nucs = _mm_cmpeq_epi8(nucleotides, _mm_set1_epi8('C'));
            __m128i c_positions = _mm_and_si128(c_nucs, all1);
            c_counts = _mm_add_epi8(c_counts, c_positions);
            __m128i g_nucs = _mm_cmpeq_epi8(nucleotides, _mm_set1_epi8('G'));
            __m128i g_positions = _mm_and_si128(g_nucs, all1);
            g_counts = _mm_add_epi8(g_counts, g_positions);
            __m128i t_nucs = _mm_cmpeq_epi8(nucleotides, _mm_set1_epi8('T'));
            __m128i t_positions = _mm_and_si128(t_nucs, all1);
            t_counts = _mm_add_epi8(t_counts, t_positions);

            /* Manual loop unrolling gives the best result here */
            staging_base_counts_ptr[0][NUCLEOTIDE_TO_INDEX[sequence_ptr[0]]] += 1;
            staging_base_counts_ptr[1][NUCLEOTIDE_TO_INDEX[sequence_ptr[1]]] += 1;
            staging_base_counts_ptr[2][NUCLEOTIDE_TO_INDEX[sequence_ptr[2]]] += 1;
            staging_base_counts_ptr[3][NUCLEOTIDE_TO_INDEX[sequence_ptr[3]]] += 1;
            staging_base_counts_ptr[4][NUCLEOTIDE_TO_INDEX[sequence_ptr[4]]] += 1;
            staging_base_counts_ptr[5][NUCLEOTIDE_TO_INDEX[sequence_ptr[5]]] += 1;
            staging_base_counts_ptr[6][NUCLEOTIDE_TO_INDEX[sequence_ptr[6]]] += 1;
            staging_base_counts_ptr[7][NUCLEOTIDE_TO_INDEX[sequence_ptr[7]]] += 1;
            staging_base_counts_ptr[8][NUCLEOTIDE_TO_INDEX[sequence_ptr[8]]] += 1;
            staging_base_counts_ptr[9][NUCLEOTIDE_TO_INDEX[sequence_ptr[9]]] += 1;
            staging_base_counts_ptr[10][NUCLEOTIDE_TO_INDEX[sequence_ptr[10]]] += 1;
            staging_base_counts_ptr[11][NUCLEOTIDE_TO_INDEX[sequence_ptr[11]]] += 1;
            staging_base_counts_ptr[12][NUCLEOTIDE_TO_INDEX[sequence_ptr[12]]] += 1;
            staging_base_counts_ptr[13][NUCLEOTIDE_TO_INDEX[sequence_ptr[13]]] += 1;
            staging_base_counts_ptr[14][NUCLEOTIDE_TO_INDEX[sequence_ptr[14]]] += 1;
            staging_base_counts_ptr[15][NUCLEOTIDE_TO_INDEX[sequence_ptr[15]]] += 1;
            sequence_ptr += sizeof(__m128i);
            staging_base_counts_ptr += sizeof(__m128i);
        }
        size_t a_bases = horizontal_add_epu8(a_counts);
        size_t c_bases = horizontal_add_epu8(c_counts);
        size_t g_bases = horizontal_add_epu8(g_counts);
        size_t t_bases = horizontal_add_epu8(t_counts);
        size_t total = a_bases + c_bases + g_bases + t_bases;
        base_counts[A] += a_bases;
        base_counts[C] += c_bases;
        base_counts[G] += g_bases;
        base_counts[T] += t_bases;
        // By substracting the ACGT bases from the length over which the 
        // count was run, we get the N bases.
        base_counts[N] += (iterations * sizeof(__m128i)) - total;
    }
    #endif

    while(sequence_ptr < sequence_end_ptr) {
        uint8_t c = *sequence_ptr;
        uint8_t c_index = NUCLEOTIDE_TO_INDEX[c];
        base_counts[c_index] += 1;
        staging_base_counts_ptr[0][c_index] += 1;
        sequence_ptr += 1; 
        staging_base_counts_ptr += 1;
    }
    uint64_t at_counts = base_counts[A] + base_counts[T];
    uint64_t gc_counts = base_counts[C] + base_counts[G];
    double gc_content_percentage = (double)gc_counts * (double)100.0 / (double)(at_counts + gc_counts);
    uint64_t gc_content_index = (uint64_t)round(gc_content_percentage);
    assert(gc_content_index >= 0);
    assert(gc_content_index <= 100);
    self->gc_content[gc_content_index] += 1;

    staging_phred_table *staging_phred_counts_ptr = self->staging_phred_counts;
    const uint8_t *qualities_ptr = qualities;
    const uint8_t *qualities_end_ptr = qualities + sequence_length;
    const uint8_t *qualities_unroll_end_ptr = qualities_end_ptr - 4;
    uint8_t phred_offset = self->phred_offset;
    double accumulated_error_rate = 0.0;
    while(qualities_ptr < qualities_unroll_end_ptr) {
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
           of out of order execution. */
        accumulated_error_rate += (
            SCORE_TO_ERROR_RATE[q0] + SCORE_TO_ERROR_RATE[q1]) + 
            (SCORE_TO_ERROR_RATE[q2] + SCORE_TO_ERROR_RATE[q3]);
        staging_phred_counts_ptr += 4;
        qualities_ptr += 4;
    }
    while(qualities_ptr < qualities_end_ptr) {
        uint8_t q = *qualities_ptr - phred_offset;    
        if (q > PHRED_MAX) {
            PyErr_Format(
                PyExc_ValueError, 
                "Not a valid phred character: %c", *qualities_ptr
            );
            return -1;
        }
        uint8_t q_index = phred_to_index(q);
        staging_phred_counts_ptr[0][q_index] += 1;
        accumulated_error_rate += SCORE_TO_ERROR_RATE[q];
        staging_phred_counts_ptr += 1;
        qualities_ptr += 1;
    }

    meta->accumulated_error_rate = accumulated_error_rate;
    double average_error_rate = accumulated_error_rate / (double)sequence_length;
    double average_phred = -10.0 * log10(average_error_rate);
    uint64_t phred_score_index = (uint64_t)round(average_phred);
    assert(phred_score_index >= 0);
    assert(phred_score_index <= PHRED_MAX);
    self->phred_scores[phred_score_index] += 1;
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
"    A FastqRecordArrayView object.\n"
);

#define QCMetrics_add_record_array_method METH_O

static PyObject * 
QCMetrics_add_record_array(QCMetrics *self, FastqRecordArrayView *record_array) 
{
    if (!FastqRecordArrayView_CheckExact(record_array)) {
        PyErr_Format(PyExc_TypeError, 
                     "record_array should be a FastqRecordArrayView object, got %s", 
                     Py_TYPE(record_array)->tp_name);
        return NULL;
    }
    Py_ssize_t number_of_records = Py_SIZE(record_array);
    struct FastqMeta *records = record_array->records;
    for (Py_ssize_t i=0; i < number_of_records; i++) {
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
"Return a array.array on the produced base count table. \n"
);

#define QCMetrics_base_count_table_method METH_NOARGS

static PyObject *
QCMetrics_base_count_table(QCMetrics *self, PyObject *Py_UNUSED(ignore))
{
    QCMetrics_flush_staging(self);
    return PythonArray_FromBuffer(
        'Q', 
        self->base_counts, 
        self->max_length *sizeof(base_table));
}

PyDoc_STRVAR(QCMetrics_phred_count_table__doc__,
"phred_table($self, /)\n"
"--\n"
"\n"
"Return a array.array on the produced phred count table. \n"
);

#define QCMetrics_phred_count_table_method METH_NOARGS

static PyObject *
QCMetrics_phred_count_table(QCMetrics *self, PyObject *Py_UNUSED(ignore))
{
    QCMetrics_flush_staging(self);
    return PythonArray_FromBuffer(
        'Q', 
        self->phred_counts, 
        self->max_length *sizeof(phred_table));
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
    QCMetrics_flush_staging(self);
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
    QCMetrics_flush_staging(self);
    return PythonArray_FromBuffer(
        'Q',
        self->phred_scores,
        sizeof(self->phred_scores)
    );
}

static PyMethodDef QCMetrics_methods[] = {
    {"add_read",  (PyCFunction)QCMetrics_add_read, 
     QCMetrics_add_read_method,  QCMetrics_add_read__doc__},
    {"add_record_array", (PyCFunction)QCMetrics_add_record_array,
     QCMetrics_add_record_array_method, QCMetrics_add_record_array__doc__},
    {"base_count_table", (PyCFunction)QCMetrics_base_count_table, 
     QCMetrics_base_count_table_method, QCMetrics_base_count_table__doc__},
    {"phred_count_table", (PyCFunction)QCMetrics_phred_count_table, 
     QCMetrics_phred_count_table_method, QCMetrics_phred_count_table__doc__},
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
    uint64_t **adapter_counter;
    PyObject *adapters;
    size_t number_of_matchers;
    MachineWordPatternMatcher *matchers;
    size_t number_of_sse2_matchers;
    #ifdef __SSE2__
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
    uint64_t **uint64_tmp = PyMem_Malloc(sizeof(uint64_t *) * number_of_adapters);
    if (uint64_tmp == NULL) {
        PyErr_NoMemory();
        goto error;
    }
    memset(uint64_tmp, 0, sizeof(uint64_t *) * number_of_adapters);
    self->adapter_counter = uint64_tmp;
    self->adapters = NULL;
    self->matchers = NULL;
    self->max_length = 0;
    self->number_of_adapters = number_of_adapters;
    self->number_of_matchers = 0;
    self->number_of_sequences = 0;
    self->number_of_sse2_matchers = 0;
    #ifdef __SSE2__
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
        uint64_t *tmp = PyMem_Realloc(self->adapter_counter[i],
                                       new_size * sizeof(uint64_t));
        if (tmp == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        self->adapter_counter[i] = tmp;
        memset(self->adapter_counter[i] + old_size, 0,
               (new_size - old_size) * sizeof(uint64_t));
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

static inline __m128i
update_adapter_count_array_sse2(
    size_t position, 
    __m128i R, 
    __m128i already_found,
    MachineWordPatternMatcherSSE2 *matcher,
    uint64_t **adapter_counter) 
{
    size_t number_of_adapters = matcher->number_of_sequences;
    for (size_t i=0; i < number_of_adapters; i++) {
        AdapterSequenceSSE2 *adapter = matcher->sequences + i;
        __m128i adapter_found_mask = adapter->found_mask;
        if (bitwise_and_nonzero_si128(adapter_found_mask, already_found)) {
            continue;
        }
        if (bitwise_and_nonzero_si128(R, adapter_found_mask)) {
            size_t found_position = position - adapter->adapter_length + 1;
            adapter_counter[adapter->adapter_index][found_position] += 1;
            // Make sure we only find the adapter once at the earliest position;
            already_found = _mm_or_si128(already_found, adapter_found_mask);
        }
    }
    return already_found;
}
#endif

static inline uint64_t update_adapter_count_array(
    size_t position,
    uint64_t R,
    uint64_t already_found,
    MachineWordPatternMatcher *matcher,
    uint64_t **adapter_counter)
{
    size_t number_of_adapters = matcher->number_of_sequences;
    for (size_t k=0; k < number_of_adapters; k++) {
        AdapterSequence *adapter = matcher->sequences + k;
        bitmask_t adapter_found_mask = adapter->found_mask;
        if (adapter_found_mask & already_found) {
            continue;
        }
        if (R & adapter_found_mask) {
            size_t found_position = position - adapter->adapter_length + 1;
            adapter_counter[adapter->adapter_index][found_position] += 1;
            // Make sure we only find the adapter once at the earliest position;
            already_found |= adapter_found_mask;
        }
    }
    return already_found;
}

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
    size_t scalar_matcher_index = 0;
    size_t vector_matcher_index = 0;
    size_t number_of_scalar_matchers = self->number_of_matchers;
    size_t number_of_vector_matchers = self->number_of_sse2_matchers;
    while(scalar_matcher_index < number_of_scalar_matchers || 
          vector_matcher_index < number_of_vector_matchers) {
        size_t remaining_scalar_matchers = number_of_scalar_matchers - scalar_matcher_index;
        size_t remaining_vector_matchers = number_of_vector_matchers - vector_matcher_index;
        
        if (remaining_vector_matchers == 0 && remaining_scalar_matchers > 0) {
            MachineWordPatternMatcher *matcher = self->matchers + scalar_matcher_index;
            bitmask_t found_mask = matcher->found_mask;
            bitmask_t init_mask = matcher->init_mask;
            bitmask_t R = 0;
            bitmask_t *bitmask = matcher->bitmasks;
            bitmask_t already_found = 0;
            scalar_matcher_index += 1;
            for (size_t pos=0; pos<sequence_length; pos++) {
                R <<= 1;
                R |= init_mask;
                uint8_t index = NUCLEOTIDE_TO_INDEX[sequence[pos]];
                R &= bitmask[index];
                if (R & found_mask) {
                    already_found = update_adapter_count_array(
                        pos, R, already_found, matcher, self->adapter_counter
                    );
                }
            }
        }
        #ifdef __SSE2__
        else if (remaining_vector_matchers == 1 && remaining_scalar_matchers == 0) {
            MachineWordPatternMatcherSSE2 *matcher = self->sse2_matchers + vector_matcher_index;
            __m128i found_mask = matcher->found_mask;
            __m128i init_mask = matcher->init_mask;
            __m128i R = _mm_setzero_si128();
            __m128i *bitmask = matcher->bitmasks;
            __m128i already_found = _mm_setzero_si128();
            vector_matcher_index += 1;
            for (size_t pos=0; pos<sequence_length; pos++) {
                R = _mm_slli_epi64(R, 1);
                R = _mm_or_si128(R, init_mask);
                uint8_t index = NUCLEOTIDE_TO_INDEX[sequence[pos]];
                __m128i mask = bitmask[index];
                R = _mm_and_si128(R, mask);
                if (bitwise_and_nonzero_si128(R, found_mask)) {
                    already_found = update_adapter_count_array_sse2(
                        pos, R, already_found, matcher, self->adapter_counter);
                }
            }
        /* In the cases below we take advantage of out of order execution on the CPU 
           by checking two matchers at the same time. Either two sse2 matchers, or 
           a bitmask_t matcher and a vector matcher. Shift-AND is a highly dependent 
           chain of actions, meaning there is no opportunity for the CPU to do two
           thing simultaneously. By doing two shift-AND routines at the same time, 
           there are two independent paths that the CPU can evaluate using out of
           order execution. This leads to significant speedups. */
        } else if (remaining_vector_matchers == 1 && remaining_scalar_matchers == 1) {
            MachineWordPatternMatcherSSE2 *vector_matcher = self->sse2_matchers + vector_matcher_index;
            MachineWordPatternMatcher *scalar_matcher = self->matchers + scalar_matcher_index;
            __m128i vector_found_mask = vector_matcher->found_mask;
            bitmask_t scalar_found_mask = scalar_matcher->found_mask;
            __m128i vector_init_mask = vector_matcher->init_mask;
            bitmask_t scalar_init_mask = scalar_matcher->init_mask;
            __m128i vector_R = _mm_setzero_si128();
            bitmask_t scalar_R = 0;
            __m128i *vector_bitmasks = vector_matcher->bitmasks;
            bitmask_t *scalar_bitmasks = scalar_matcher->bitmasks;
            __m128i vector_already_found = _mm_setzero_si128();
            bitmask_t scalar_already_found = 0;
            vector_matcher_index += 1;
            scalar_matcher_index += 1;
            for (size_t pos=0; pos<sequence_length; pos++) {
                vector_R = _mm_slli_epi64(vector_R, 1);
                scalar_R <<= 1;
                vector_R = _mm_or_si128(vector_R, vector_init_mask);
                scalar_R |= scalar_init_mask;
                uint8_t index = NUCLEOTIDE_TO_INDEX[sequence[pos]];
                scalar_R &= scalar_bitmasks[index];
                __m128i vector_mask = vector_bitmasks[index];
                vector_R = _mm_and_si128(vector_R, vector_mask);
                if (bitwise_and_nonzero_si128(vector_R, vector_found_mask)) {
                    vector_already_found = update_adapter_count_array_sse2(
                        pos, vector_R, vector_already_found, vector_matcher, 
                        self->adapter_counter);
                }
                if (scalar_R & scalar_found_mask) {
                    scalar_already_found = update_adapter_count_array(
                        pos, scalar_R, scalar_already_found, scalar_matcher,
                        self->adapter_counter
                    );
                }
            }
        } else if (remaining_vector_matchers > 1) {
            MachineWordPatternMatcherSSE2 *matcher1 = self->sse2_matchers + vector_matcher_index;
            MachineWordPatternMatcherSSE2 *matcher2 = self->sse2_matchers + vector_matcher_index + 1;
            __m128i found_mask1 = matcher1->found_mask;
            __m128i found_mask2 = matcher2->found_mask;
            __m128i init_mask1 = matcher1->init_mask;
            __m128i init_mask2 = matcher2->init_mask;
            __m128i R1 = _mm_setzero_si128();
            __m128i R2 = _mm_setzero_si128();
            __m128i *bitmasks1 = matcher1->bitmasks;
            __m128i *bitmasks2 = matcher2->bitmasks;
            __m128i already_found1 = _mm_setzero_si128();
            __m128i already_found2 = _mm_setzero_si128();
            vector_matcher_index += 2;            
            for (size_t pos=0; pos<sequence_length; pos++) {
                R1 = _mm_slli_epi64(R1, 1);
                R2 = _mm_slli_epi64(R2, 1);
                R1 = _mm_or_si128(R1, init_mask1);
                R2 = _mm_or_si128(R2, init_mask2);
                uint8_t index = NUCLEOTIDE_TO_INDEX[sequence[pos]];
                __m128i mask1 = bitmasks1[index];
                __m128i mask2 = bitmasks2[index];
                R1 = _mm_and_si128(R1, mask1);
                R2 = _mm_and_si128(R2, mask2);
                if (bitwise_and_nonzero_si128(R1, found_mask1)) {
                    already_found1 = update_adapter_count_array_sse2(
                        pos, R1, already_found1, matcher1, self->adapter_counter);
                }
                if (bitwise_and_nonzero_si128(R2, found_mask2)) {
                    already_found2 = update_adapter_count_array_sse2(
                        pos, R2, already_found2, matcher2, self->adapter_counter);
                }
            }
        }
        #endif
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
"    A FastqRecordArrayView object.\n"
);

#define AdapterCounter_add_record_array_method METH_O

static PyObject * 
AdapterCounter_add_record_array(AdapterCounter *self, FastqRecordArrayView *record_array) 
{
    if (!FastqRecordArrayView_CheckExact(record_array)) {
        PyErr_Format(PyExc_TypeError, 
                     "record_array should be a FastqRecordArrayView object, got %s", 
                     Py_TYPE(record_array)->tp_name);
        return NULL;
    }
    Py_ssize_t number_of_records = Py_SIZE(record_array);
    struct FastqMeta *records = record_array->records;
    for (Py_ssize_t i=0; i < number_of_records; i++) {
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
"Return the counts as a list of tuples. Each tuple contains the adapter, \n"
"and an array.array counts per position. \n"
);

# define AdapterCounter_get_counts_method METH_NOARGS

static PyObject *
AdapterCounter_get_counts(AdapterCounter *self, PyObject *Py_UNUSED(ignore))
{
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
            self->max_length * sizeof(uint64_t)
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
    {"add_record_array", (PyCFunction)AdapterCounter_add_record_array,
     AdapterCounter_add_record_array_method, AdapterCounter_add_record_array__doc__},
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

static PyTypeObject Adapteruint64_type = {
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
PerTileQuality_dealloc(PerTileQuality *self) {
    Py_XDECREF(self->skipped_reason);
    for (size_t i=0; i < self->number_of_tiles; i++) {
        TileQuality tile_qual = self->tile_qualities[i];
        PyMem_Free(tile_qual.length_counts);
        PyMem_Free(tile_qual.total_errors);
    }
    PyMem_Free(self->tile_qualities);
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
    TileQuality *new_qualities = PyMem_Realloc(
        self->tile_qualities, highest_tile * sizeof(TileQuality));
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
    for (size_t i=0; i<number_of_tiles; i++) {
        TileQuality *tile_quality = tile_qualities + i;
        if (tile_quality->length_counts == NULL && tile_quality->total_errors == NULL) {
            continue;
        }
        uint64_t *length_counts = PyMem_Realloc(tile_quality->length_counts, new_length *sizeof(uint64_t));
        double *total_errors = PyMem_Realloc(tile_quality->total_errors, new_length *sizeof(double));
        
        if (length_counts == NULL || total_errors == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        memset(length_counts + old_length, 0, (new_length - old_length) * sizeof(uint64_t));
        memset(total_errors + old_length, 0, (new_length - old_length) * sizeof(double));
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
static
Py_ssize_t illumina_header_to_tile_id(const uint8_t *header, size_t header_length) {

    /* The following link contains the header format:
       https://support.illumina.com/help/BaseSpace_OLH_009008/Content/Source/Informatics/BS/FileFormat_FASTQ-files_swBS.htm
       It reports the following format:
       @<instrument>:<run number>:<flowcell ID>:<lane>:<tile>:<x-pos>:<y-pos>:<UMI> <read>:<is filtered>:<control number>:<index>
       The tile ID is after the fourth colon.
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
    while(cursor < header_end) {
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
    const uint8_t *header = record_start + 1;
    size_t header_length = meta->name_length;
    const uint8_t *qualities = record_start + meta->qualities_offset;
    size_t sequence_length = meta->sequence_length;
    uint8_t phred_offset = self->phred_offset;

    Py_ssize_t tile_id = illumina_header_to_tile_id(header, header_length);
    if (tile_id == -1) {
        PyObject *header_obj = PyUnicode_DecodeASCII((const char *)header, header_length, NULL);
        if (header_obj == NULL) {
            return -1;
        } 
        self->skipped_reason = PyUnicode_FromFormat(
            "Can not parse header: %R",
            header_obj);
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
        uint64_t *length_counts = PyMem_Malloc(self->max_length *sizeof(uint64_t));
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
    double *total_errors = tile_quality->total_errors;
    double *error_cursor = total_errors;
    const uint8_t *qualities_end = qualities + sequence_length;
    const uint8_t *qualities_ptr = qualities;
    #ifdef __SSE2__
    const uint8_t *qualities_vec_end = qualities_end - sizeof(__m128i);
    while (qualities_ptr < qualities_vec_end) {
        __m128i phreds = _mm_loadu_si128((__m128i *)qualities_ptr);
        __m128i too_low_phreds = _mm_cmplt_epi8(phreds, _mm_set1_epi8(phred_offset));
        __m128i too_high_phreds = _mm_cmpgt_epi8(phreds, _mm_set1_epi8(126));
        if (_mm_movemask_epi8(_mm_or_si128(too_low_phreds, too_high_phreds))) {
            /* Find the culprit in the non-vectorized loop*/
            break;
        }
        /* Since the actions are independent the compiler will unroll this loop */
        for (size_t i=0; i<16; i += 2) {
            __m128d current_errors = _mm_loadu_pd(error_cursor + i);
            __m128d sequence_errors = _mm_setr_pd(
                SCORE_TO_ERROR_RATE[qualities_ptr[i] - phred_offset],
                SCORE_TO_ERROR_RATE[qualities_ptr[i+1] - phred_offset]
            );
            _mm_storeu_pd(error_cursor + i, _mm_add_pd(current_errors, sequence_errors));
        }
        error_cursor += sizeof(__m128i);
        qualities_ptr += sizeof(__m128i);
    }
    #endif
    while (qualities_ptr < qualities_end) {
        uint8_t q = *qualities_ptr - phred_offset;
        if (q > PHRED_MAX) {
            PyErr_Format(
                PyExc_ValueError,
                "Not a valid phred character: %c", *qualities_ptr
            );
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
"    A FastqRecordView object.\n"
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
"    A FastqRecordArrayView object.\n"
);

#define PerTileQuality_add_record_array_method METH_O

static PyObject * 
PerTileQuality_add_record_array(PerTileQuality *self, FastqRecordArrayView *record_array) 
{
    if (self->skipped) {
        Py_RETURN_NONE;
    }
    if (!FastqRecordArrayView_CheckExact(record_array)) {
        PyErr_Format(PyExc_TypeError, 
                     "record_array should be a FastqRecordArrayView object, got %s", 
                     Py_TYPE(record_array)->tp_name);
        return NULL;
    }
    Py_ssize_t number_of_records = Py_SIZE(record_array);
    struct FastqMeta *records = record_array->records;
    for (Py_ssize_t i=0; i < number_of_records; i++) {
        if (PerTileQuality_add_meta(self, records + i) != 0) {
           return NULL;
        }
    }
    Py_RETURN_NONE;
}


PyDoc_STRVAR(PerTileQuality_get_tile_averages__doc__,
"get_tile_averages($self, /)\n"
"--\n"
"\n"
"Get a list of tuples with the tile IDs and a list of their averages. \n"
);

#define PerTileQuality_get_tile_averages_method METH_NOARGS

static PyObject *
PerTileQuality_get_tile_averages(PerTileQuality *self, PyObject *Py_UNUSED(ignore))
{
    TileQuality *tile_qualities = self->tile_qualities;
    size_t maximum_tile = self->number_of_tiles;
    size_t tile_length = self->max_length;
    PyObject *result = PyList_New(0);
    if (result == NULL) {
        return PyErr_NoMemory();
    }

    for (size_t i=0; i<maximum_tile; i++) {
        TileQuality *tile_quality = tile_qualities + i;
        double *total_errors = tile_quality->total_errors;
        uint64_t *length_counts = tile_quality->length_counts;
        if (length_counts == NULL && total_errors == NULL) {
            continue;
        }
        PyObject *entry = PyTuple_New(2);
        PyObject *tile_id = PyLong_FromSize_t(i);
        PyObject *averages_list = PyList_New(tile_length);
        if (entry == NULL || tile_id == NULL || averages_list == NULL) {
            Py_DECREF(result);
            return PyErr_NoMemory();
        }
        
        /* Work back from the lenght counts. If we have 200 reads total and a
           100 are length 150 and a 100 are length 120. This means we have 
           a 100 bases at each position 120-150 and 200 bases at 0-120. */
        uint64_t total_bases = 0;
        for (Py_ssize_t j=tile_length - 1; j >= 0; j -= 1) {
            total_bases += length_counts[j];
            double error_count = total_errors[j];
            double average = error_count / (double)total_bases;
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

PyDoc_STRVAR(PerTileQuality_get_tile_counts__doc__,
"get_tile_counts($self, /)\n"
"--\n"
"\n"
"Get a list of tuples with the tile IDs and a list of their summed errors and\n"
"a list of their counts. \n"
);

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

    for (size_t i=0; i<maximum_tile; i++) {
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
        if (entry == NULL || tile_id == NULL || summed_error_list == NULL || count_list == NULL) {
            Py_DECREF(result);
            return PyErr_NoMemory();
        }
        /* Work back from the lenght counts. If we have 200 reads total and a
           100 are length 150 and a 100 are length 120. This means we have 
           a 100 bases at each position 120-150 and 200 bases at 0-120. */
        uint64_t total_bases = 0;
        for (Py_ssize_t j=tile_length - 1; j >= 0; j -= 1) {
            total_bases += length_counts[j];
            PyObject *summed_error_obj = PyFloat_FromDouble(total_errors[j]);
            PyObject *count_obj = PyLong_FromUnsignedLongLong(total_bases);
            if (summed_error_obj == NULL || count_obj == NULL) {
                Py_DECREF(result);
                return PyErr_NoMemory();
            }
            PyList_SET_ITEM(summed_error_list, j, summed_error_obj);
            PyList_SET_ITEM(count_list, j, count_obj);
        }
        PyTuple_SET_ITEM(entry, 0, tile_id);
        PyTuple_SET_ITEM(entry, 1, summed_error_list);
        PyTuple_SET_ITEM(entry, 2, count_list);
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
     PerTileQuality_add_record_array_method, PerTileQuality_add_record_array__doc__},
    {"get_tile_averages", (PyCFunction)PerTileQuality_get_tile_averages,
     PerTileQuality_get_tile_averages_method, 
     PerTileQuality_get_tile_averages__doc__},
     {"get_tile_counts", (PyCFunction)PerTileQuality_get_tile_counts,
     PerTileQuality_get_tile_counts_method,
     PerTileQuality_get_tile_counts__doc__},
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

/**********************
 * TWOBIT CONVERSIONS *
 **********************/

/* To be used in the sequence duplication part */

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

#define TWOBIT_UNKNOWN_CHAR -1
#define TWOBIT_N_CHAR -2 
#define TWOBIT_SUCCESS 0

static uint64_t reverse_complement_kmer(uint64_t kmer, uint64_t k) {
    // Invert all the bits, with 0,1,2,3 == A,C,G,T this automatically is the
    // complement. 
    uint64_t comp = ~kmer; 
    // Progressively swap all the twobits inplace.
    uint64_t revcomp = (comp << 32) | (comp >> 32);
    revcomp = ((revcomp & 0xFFFF0000FFFF0000ULL) >> 16) | 
              ((revcomp & 0x0000FFFF0000FFFFULL) << 16);
    revcomp = ((revcomp & 0xFF00FF00FF00FF00ULL) >> 8) | 
              ((revcomp & 0x00FF00FF00FF00FFULL) << 8);
    revcomp = ((revcomp & 0xF0F0F0F0F0F0F0F0ULL) >> 4) | 
              ((revcomp & 0x0F0F0F0F0F0F0F0FULL) << 4);
    revcomp = ((revcomp & 0xCCCCCCCCCCCCCCCCULL) >> 2) | 
              ((revcomp & 0x3333333333333333ULL) << 2);
    // If k < 32, the empty twobit slots will have ended up at the least 
    // significant bits. Use a shift to move them to the highest bits again.
    return revcomp >> (64 - (k *2));
}

static int64_t sequence_to_canonical_kmer(uint8_t *sequence, uint64_t k) {
    uint64_t kmer = 0;
    size_t all_nucs = 0;
    Py_ssize_t i=0;
    Py_ssize_t vector_end = k - 4;
    for (i=0; i<vector_end; i+=4) {
        size_t nuc0 = NUCLEOTIDE_TO_TWOBIT[sequence[i]];
        size_t nuc1 = NUCLEOTIDE_TO_TWOBIT[sequence[i+1]];
        size_t nuc2 = NUCLEOTIDE_TO_TWOBIT[sequence[i+2]];
        size_t nuc3 = NUCLEOTIDE_TO_TWOBIT[sequence[i+3]];
        all_nucs |= (nuc0 | nuc1 | nuc2 | nuc3);
        uint64_t kchunk = ((nuc0 << 6) | (nuc1 << 4) | (nuc2 << 2) | (nuc3));
        kmer <<= 8;
        kmer |= kchunk;
    }
    for (i=i; i<(Py_ssize_t)k; i++) {
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

static void kmer_to_sequence(uint64_t kmer, size_t k, uint8_t *sequence) {
    static uint8_t nucs[4] = {'A', 'C', 'G', 'T'};
    for (size_t i=k; i>0; i-=1) {
        size_t nuc = kmer & 0b11;
        sequence[i - 1] = nucs[nuc];
        kmer >>= 2;
    }
}

/*************************
 * SEQUENCE DUPLICATION *
 *************************/

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

typedef struct _SequenceDuplicationStruct {
    PyObject_HEAD 
    size_t fragment_length;
    uint64_t number_of_sequences;
    uint64_t sampled_sequences;
    uint64_t hash_table_size;
    uint64_t *hashes; 
    uint32_t *counts;
    uint64_t max_unique_fragments;
    uint64_t number_of_unique_fragments;
    uint64_t total_fragments;
    size_t sample_every;
} SequenceDuplication;

static void 
SequenceDuplication_dealloc(SequenceDuplication *self)
{
    PyMem_Free(self->hashes);
    PyMem_Free(self->counts);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
SequenceDuplication__new__(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    Py_ssize_t max_unique_fragments = DEFAULT_MAX_UNIQUE_FRAGMENTS;
    Py_ssize_t fragment_length = DEFAULT_FRAGMENT_LENGTH;
    Py_ssize_t sample_every = DEFAULT_UNIQUE_SAMPLE_EVERY;
    static char *kwargnames[] = {"max_unique_fragments", "fragment_length", 
                                 "sample_every", NULL};
    static char *format = "|nnn:SequenceDuplication";
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, kwargnames,
            &max_unique_fragments, &fragment_length, &sample_every)) {
        return NULL;
    }
    if (max_unique_fragments < 1) {
        PyErr_Format(
            PyExc_ValueError, 
            "max_unique_fragments should be at least 1, got: %zd", 
            max_unique_fragments);
        return NULL;
    }
    if ((fragment_length & 1) == 0 || fragment_length > 31 || fragment_length < 3) {
        PyErr_Format(
            PyExc_ValueError,
            "fragment_length must be between 3 and 31 and be an uneven number, got: %zd", 
            fragment_length
        );
        return NULL;
    }
    if (sample_every < 1) {
        PyErr_Format(
            PyExc_ValueError,
            "sample_every must be 1 or greater. Got %zd", 
            sample_every
        );
        return NULL;
    }
    /* If size is a power of 2, the modulo HASH_TABLE_SIZE can be optimised to a
       bitwise AND. Using 1.5 times as a base we ensure that the hashtable is
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
    SequenceDuplication *self = PyObject_New(SequenceDuplication, type);
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
    self->hashes = hashes;
    self->counts = counts;
    self->sample_every = sample_every;
    return (PyObject *)self;
}

static void
Sequence_duplication_insert_hash(SequenceDuplication *self, uint64_t hash) 
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
        } else if (hash_entry == hash) {
            counts[index] +=1;
            break;
        }
        index += 1;
        /* Make sure the index round trips when it reaches hash_table_size.*/
        index &= hash_to_index_int;
    }
}

static int
SequenceDuplication_add_meta(SequenceDuplication *self, struct FastqMeta *meta)
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
    Py_ssize_t mid_point = (sequence_length + 1) / 2;
    Py_ssize_t total_fragments = (sequence_length + fragment_length - 1) / fragment_length;
    Py_ssize_t from_mid_point_fragments = total_fragments / 2;
    Py_ssize_t mid_point_start = sequence_length - (from_mid_point_fragments * fragment_length);
    bool warn_unknown = false;
    // Save all fragments starting from 0 and up to the midpoint.
    for (Py_ssize_t i = 0; i < mid_point; i += fragment_length) {
        int64_t kmer = sequence_to_canonical_kmer(sequence + i, fragment_length);
        if (kmer < 0) {
            if (kmer == TWOBIT_UNKNOWN_CHAR) {
                warn_unknown = true;
            }
            continue;
        }
        fragments += 1;
        uint64_t hash = wanghash64(kmer);
        Sequence_duplication_insert_hash(self, hash);
    }
    // Save all subsequences of length k starting from the end until the point 
    // where the previous loop has saved the sequences. There might be slight 
    // overlap in the middle..
    for (Py_ssize_t i = mid_point_start; i < sequence_length; i += fragment_length) {
        int64_t kmer = sequence_to_canonical_kmer(sequence + i, fragment_length);
        if (kmer < 0) {
            if (kmer == TWOBIT_UNKNOWN_CHAR) {
                warn_unknown = true;
            }
            continue;
        }
        fragments += 1;
        uint64_t hash = wanghash64(kmer);
        Sequence_duplication_insert_hash(self, hash);
    }
    if (warn_unknown) {
        PyErr_WarnFormat(
            PyExc_UserWarning, 
            1,
            "Sequence contains a chacter that is not A, C, G, T or N: %R", 
            PyUnicode_DecodeASCII((char *)sequence, sequence_length, NULL)
        );
    }
    self->total_fragments += fragments;
    return 0;
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
    if (SequenceDuplication_add_meta(self, &read->meta) !=0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(SequenceDuplication_add_record_array__doc__,
"add_record_array($self, record_array, /)\n"
"--\n"
"\n"
"Add a record_array to the duplication module. \n"
"\n"
"  record_array\n"
"    A FastqRecordArrayView object.\n"
);

#define SequenceDuplication_add_record_array_method METH_O

static PyObject * 
SequenceDuplication_add_record_array(
    SequenceDuplication *self, FastqRecordArrayView *record_array) 
{
    if (!FastqRecordArrayView_CheckExact(record_array)) {
        PyErr_Format(PyExc_TypeError, 
                     "record_array should be a FastqRecordArrayView object, got %s", 
                     Py_TYPE(record_array)->tp_name);
        return NULL;
    }
    Py_ssize_t number_of_records = Py_SIZE(record_array);
    struct FastqMeta *records = record_array->records;
    for (Py_ssize_t i=0; i < number_of_records; i++) {
        if (SequenceDuplication_add_meta(self, records + i) !=0) {
           return NULL;
        }
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
    uint64_t *hashes = self->hashes;
    uint32_t *counts = self->counts;
    uint64_t hash_table_size = self->hash_table_size;
    Py_ssize_t fragment_length = self->fragment_length;
    for (size_t i=0; i < hash_table_size; i+=1) {
        uint64_t entry_hash = hashes[i];
        if  (entry_hash == 0) {
            continue;
        }
        PyObject *count_obj = PyLong_FromUnsignedLong(counts[i]);
        if (count_obj == NULL) {
            goto error;
        }
        PyObject *key = PyUnicode_New(fragment_length, 127);
        if (key == NULL) {
            goto error;
        }
        uint64_t kmer = wanghash64_inverse(entry_hash);
        kmer_to_sequence(kmer, fragment_length, PyUnicode_DATA(key));
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
"Return a list of tuples with the count, fraction and the sequence. The list is "
"sorted in reverse order with the most common sequence on top.\n"
"\n"
"  threshold_fraction\n"
"    The fraction at which a sequence is considered overrepresented.\n"
"  min_threshold\n"
"    the minimum threshold to uphold. Overrides the minimum number based on "
"    the threshold_fraction if it is higher. Useful for files with very low " 
"    numbers of sequences."
"  max_threshold\n"
"    the maximum threshold to uphold. Overrides the minimum number based on "
"    the threshold_fraction if it is lower. Useful for files with very high " 
"    numbers of sequences."
);

#define SequenceDuplication_overrepresented_sequences_method METH_VARARGS | METH_KEYWORDS

static PyObject *
SequenceDuplication_overrepresented_sequences(SequenceDuplication *self, 
                                              PyObject *args, PyObject *kwargs)
{
    double threshold = 0.0001;  // 0.01 %
    Py_ssize_t min_threshold = 1;
    Py_ssize_t max_threshold = PY_SSIZE_T_MAX;
    static char *kwargnames[] = {"threshold_fraction", 
                                 "min_threshold", 
                                 "max_threshold",
                                  NULL};
    static char *format = "|dnn:SequenceDuplication.overrepresented_sequences";
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, kwargnames, 
        &threshold, &min_threshold, &max_threshold)) {
        return NULL;
    }
    if ((threshold < 0.0) || (threshold > 1.0)) {
        // PyErr_Format has no direct way to represent floats
        PyObject *threshold_obj = PyFloat_FromDouble(threshold);
        PyErr_Format(
            PyExc_ValueError, 
            "threshold_fraction must be between 0.0 and 1.0 got, %R", threshold_obj, 
            threshold);
        Py_XDECREF(threshold_obj);
        return NULL;
    }
    if (min_threshold < 1) {
        PyErr_Format(
            PyExc_ValueError, 
            "min_threshold must be at least 1, got %zd", min_threshold);
        return NULL;
    }
    if (max_threshold < 1) {
        PyErr_Format(
            PyExc_ValueError,
            "max_threshold must be at least 1, got %zd", max_threshold);
        return NULL;
    }
    if (max_threshold < min_threshold) {
        PyErr_Format(
            PyExc_ValueError,
            "max_threshold (%zd) must be greater than min_threshold (%zd)",
            max_threshold, min_threshold
        );
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
    for (size_t i=0; i < hash_table_size; i+=1) {
        uint32_t count = counts[i];
        if (count >= minimum_hits) {
            uint64_t entry_hash = hashes[i];
            uint64_t kmer = wanghash64_inverse(entry_hash);
            PyObject *sequence_obj = PyUnicode_New(fragment_length, 127);
            if (sequence_obj == NULL) {
                goto error;
            }
            kmer_to_sequence(kmer, fragment_length, PyUnicode_DATA(sequence_obj));
            PyObject *entry_tuple = Py_BuildValue(
                "(KdN)",
                count,
                (double)((double)count / (double)sampled_sequences),
                sequence_obj);
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
"duplication_counts($self)\n"
"--\n"
"\n"
"Return a array.array with only the counts.\n"
);

#define SequenceDuplication_duplication_counts_method METH_NOARGS

static PyObject *
SequenceDuplication_duplication_counts(SequenceDuplication *self, 
									   PyObject *Py_UNUSED(ignore))
{
    uint64_t number_of_uniques = self->number_of_unique_fragments;
	uint64_t *counts = PyMem_Calloc(number_of_uniques, sizeof(uint64_t));
    if (counts == NULL) {
        return PyErr_NoMemory();
    }
    uint32_t *counters = self->counts;
    size_t count_index = 0;
    size_t hash_table_size = self->hash_table_size;

    for (size_t i=0; i < hash_table_size; i+=1) {
        uint32_t count = counters[i];
        if (count != 0) {
            counts[count_index] = count;
            count_index += 1;
        }
    }
    PyObject *result = PythonArray_FromBuffer('Q', counts, number_of_uniques * sizeof(uint64_t));
    PyMem_Free(counts);
    return result;
}

static PyMethodDef SequenceDuplication_methods[] = {
    {"add_read", (PyCFunction)SequenceDuplication_add_read, 
     SequenceDuplication_add_read_method, 
     SequenceDuplication_add_read__doc__},
    {"add_record_array", (PyCFunction)SequenceDuplication_add_record_array,
     SequenceDuplication_add_record_array_method, 
     SequenceDuplication_add_record_array__doc__},
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
     "The total number of sequences submitted."},
    {"sampled_sequences", T_ULONGLONG,
     offsetof(SequenceDuplication, sampled_sequences), READONLY,
     "The total number of sequences that were analysed."},
    {"collected_unique_fragments", T_ULONGLONG,
      offsetof(SequenceDuplication, number_of_unique_fragments), READONLY,
      "The number of unique fragments collected."},
    {"max_unique_fragments", T_ULONGLONG,
      offsetof(SequenceDuplication, max_unique_fragments), READONLY,
      "The maximum number of unique sequences stored in the object."
    }, 
    {"fragment_length", T_BYTE, offsetof(SequenceDuplication, fragment_length), READONLY,
     "The length of the sampled sequences"},
    {"sample_every", T_PYSSIZET, offsetof(SequenceDuplication, sample_every), 
     READONLY, "One in this many reads is sampled"},
	 {"total_fragments", T_ULONGLONG, offsetof(SequenceDuplication, total_fragments),
     READONLY, "Total number of fragments."},
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

/*******************
 * DEDUP ESTIMATOR *
 *******************/
/*
Based on the following paper:
Estimating Duplication by Content-based Sampling
Fei Xie, Michael Condict, Sandip Shete
https://www.usenix.org/system/files/conference/atc13/atc13-xie.pdf
*/

// 2 ** 21 * 12 is 24MB which balloons to 48MB when creating a new table.
// This allows storing up to 1.46 million sequences which leads to quite
// accurate results.
#define DEFAULT_DEDUP_HASH_TABLE_SIZE_BITS 21

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
    struct EstimatorEntry *hash_table;
} DedupEstimator;

static void 
DedupEstimator_dealloc(DedupEstimator *self) {
    PyMem_Free(self->hash_table);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
DedupEstimator__new__(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    Py_ssize_t hash_table_size_bits = DEFAULT_DEDUP_HASH_TABLE_SIZE_BITS;
    static char *kwargnames[] = {"hash_table_size_bits", NULL};
    static char *format = "|n:DedupEstimator";
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, kwargnames,
            &hash_table_size_bits)) {
        return NULL;
    }
    if (hash_table_size_bits < 8 || hash_table_size_bits > 58) {
        PyErr_Format(
            PyExc_ValueError, 
            "hash_table_size_bits must be between 8 and 58, not %zd",
            hash_table_size_bits
        );
        return NULL;
    }
    size_t hash_table_size = 1ULL << hash_table_size_bits;
    struct EstimatorEntry *hash_table = PyMem_Calloc(hash_table_size, sizeof(struct EstimatorEntry));
    if (hash_table == NULL) {
        return PyErr_NoMemory();
    }
    DedupEstimator *self = PyObject_New(DedupEstimator, type);
    if (self == NULL) {
        PyMem_Free(hash_table);
        return PyErr_NoMemory();
    }
    self->hash_table_size = hash_table_size;
    // Get about 70% occupancy max
    self->max_stored_entries = (hash_table_size * 7) / 10;
    self->hash_table = hash_table;
    self->modulo_bits = 1;
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
    struct EstimatorEntry *new_hash_table = PyMem_Calloc(hash_table_size, sizeof(struct EstimatorEntry));
    if (new_hash_table == NULL) {
        PyErr_NoMemory();
        return -1;
    } 

    for (size_t i=0; i < hash_table_size; i++) {
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

/* 
Avoid the beginning and end of the sequence by at most 64 bp to avoid
any adapters. Take the 8 bp after the start offset and the 8 bp before 
the end offset. This creates a small 16 bp fingerprint. Hash it using 
MurmurHash. 16 bp is small and therefore relatively insensitive to 
sequencing errors while still offering 4^16 or 4 billion distinct 
fingerprints. 
*/
#define FINGERPRINT_MAX_OFFSET 64
#define FINGERPRINT_LENGTH 16

static int 
DedupEstimator_add_sequence_ptr(DedupEstimator *self, 
                               uint8_t *sequence, size_t sequence_length) 
{

    uint64_t hash;
    if (sequence_length < 16) {
        hash = MurmurHash3_x64_64(sequence, sequence_length, 0);
    } else {
        uint64_t seed = sequence_length >> 6;
        uint8_t fingerprint[FINGERPRINT_LENGTH];
        size_t remainder = sequence_length - FINGERPRINT_LENGTH;
        size_t offset = Py_MIN(remainder / 2, FINGERPRINT_MAX_OFFSET);
        memcpy(fingerprint, sequence + offset, FINGERPRINT_LENGTH / 2);
        memcpy(fingerprint + (FINGERPRINT_LENGTH / 2), 
               sequence + sequence_length - (offset + (FINGERPRINT_LENGTH / 2)), 
               (FINGERPRINT_LENGTH / 2));
        hash = MurmurHash3_x64_64(fingerprint, FINGERPRINT_LENGTH, seed);
    }
    size_t modulo_bits = self->modulo_bits;
    size_t ignore_mask = (1ULL << modulo_bits) - 1;
    if (hash & ignore_mask) {
        return 0;
    }
    size_t hash_table_size = self->hash_table_size;
    if (self->stored_entries >= self->max_stored_entries) {
        if (DedupEstimator_increment_modulo(self) != 0) {
            return - 1;
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

PyDoc_STRVAR(DedupEstimator_add_record_array__doc__,
"add_record_array($self, record_array, /)\n"
"--\n"
"\n"
"Add a record_array to the deduplication estimator. \n"
"\n"
"  record_array\n"
"    A FastqRecordArrayView object.\n"
);

#define DedupEstimator_add_record_array_method METH_O

static PyObject *
DedupEstimator_add_record_array(DedupEstimator *self, FastqRecordArrayView *record_array) 
{
    if (!FastqRecordArrayView_CheckExact(record_array)) {
        PyErr_Format(PyExc_TypeError, 
                     "record_array should be a FastqRecordArrayView object, got %s", 
                     Py_TYPE(record_array)->tp_name);
        return NULL;
    }
    Py_ssize_t number_of_records = Py_SIZE(record_array);
    struct FastqMeta *records = record_array->records;
    for (Py_ssize_t i=0; i < number_of_records; i++) {
        struct FastqMeta *meta = records + i;
        uint8_t *sequence = meta->record_start + meta->sequence_offset;
        size_t sequence_length = meta->sequence_length;
        if (DedupEstimator_add_sequence_ptr(self, sequence, sequence_length) != 0) {
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
"    An ASCII string.\n"
);

#define DedupEstimator_add_sequence_method METH_O

static PyObject *
DedupEstimator_add_sequence(DedupEstimator *self, PyObject *sequence) 
{
    if (!PyUnicode_CheckExact(sequence)) {
        PyErr_Format(PyExc_TypeError, 
                     "sequence should be a str object, got %s", 
                     Py_TYPE(sequence)->tp_name);
        return NULL;
    }
    if (!PyUnicode_IS_COMPACT_ASCII(sequence)) {
        PyErr_SetString(
            PyExc_ValueError, 
            "sequence should consist only of ASCII characters.");
        return NULL;
    }
    if (DedupEstimator_add_sequence_ptr(
            self, PyUnicode_DATA(sequence), PyUnicode_GET_LENGTH(sequence)) != 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}


PyDoc_STRVAR(DedupEstimator_duplication_counts__doc__,
"duplication_counts($self)\n"
"--\n"
"\n"
"Return a array.array with only the counts. \n"
);

#define DedupEstimator_duplication_counts_method METH_NOARGS

static PyObject *
DedupEstimator_duplication_counts(DedupEstimator *self, PyObject *Py_UNUSED(ignore)) 
{
    size_t tracked_sequences = self->stored_entries;
    uint64_t *counts = PyMem_Calloc(tracked_sequences, sizeof(uint64_t));
    if (counts == NULL) {
        return PyErr_NoMemory();
    }
    struct EstimatorEntry *hash_table = self->hash_table;
    size_t hash_table_size = self->hash_table_size;
    size_t count_index = 0;
    for (size_t i=0; i < hash_table_size; i++) {
        struct EstimatorEntry *entry = hash_table + i;
        uint64_t count = entry->count;
        if (count == 0) {
            continue;
        }
        counts[count_index] = count;
        count_index += 1;
    }
    PyObject *result = PythonArray_FromBuffer('Q', counts, tracked_sequences * sizeof(uint64_t));
    PyMem_Free(counts);
    return result;
} 

static PyMethodDef DedupEstimator_methods[] = {
    {"add_record_array", (PyCFunction)DedupEstimator_add_record_array, 
     DedupEstimator_add_record_array_method, DedupEstimator_add_record_array__doc__},
    {"add_sequence", (PyCFunction)DedupEstimator_add_sequence, 
     DedupEstimator_add_sequence_method, DedupEstimator_add_sequence__doc__},
    {"duplication_counts", (PyCFunction)DedupEstimator_duplication_counts, 
     DedupEstimator_duplication_counts_method, DedupEstimator_duplication_counts__doc__},
    {NULL},
};

static PyMemberDef DedupEstimator_members[] = {
    {"_modulo_bits", T_ULONGLONG, offsetof(DedupEstimator, modulo_bits), 
     READONLY, NULL},
    {"_hash_table_size", T_ULONGLONG, offsetof(DedupEstimator, hash_table_size), 
     READONLY, NULL},
    {"tracked_sequences", T_ULONGLONG, offsetof(DedupEstimator, stored_entries), 
     READONLY, NULL},
    {NULL},
};

static PyTypeObject DedupEstimator_Type = {
    .tp_name = "_qc.DedupEstimator",
    .tp_basicsize = sizeof(DedupEstimator),
    .tp_dealloc = (destructor)(DedupEstimator_dealloc),
    .tp_new = (newfunc)DedupEstimator__new__,
    .tp_methods = DedupEstimator_methods,
    .tp_members = DedupEstimator_members,
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
};

typedef struct {
    PyObject_HEAD
    struct NanoInfo info;
} NanoporeReadInfo;

static PyObject *
NanoporeReadInfo_get_start_time(NanoporeReadInfo *self, void *closure) {
    return PyLong_FromLong(self->info.start_time);
}    
static PyObject *
NanoporeReadInfo_get_channel_id(NanoporeReadInfo *self, void *closure) {
    return PyLong_FromLong(self->info.channel_id);
}
static PyObject *
NanoporeReadInfo_get_length(NanoporeReadInfo *self, void *closure) {
    return PyLong_FromUnsignedLong(self->info.length);
}
static PyObject *
NanoporeReadInfo_get_cumulative_error_rate(NanoporeReadInfo *self, void *closure) {
    return PyFloat_FromDouble(self->info.cumulative_error_rate);
}
static PyObject *
NanoporeReadInfo_get_duration(NanoporeReadInfo *self, void *closure) {
    return PyFloat_FromDouble((double)self->info.duration);
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
    {NULL},
};

static PyTypeObject NanoporeReadInfo_Type = {
    .tp_name = "_qc.NanoporeReadInfo",
    .tp_basicsize = sizeof(NanoporeReadInfo),
    .tp_dealloc = (destructor)PyObject_Del,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = NanoporeReadInfo_properties,
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

static void NanoStats_dealloc(NanoStats *self) {
    PyMem_Free(self->nano_infos);
    Py_XDECREF(self->skipped_reason);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

typedef struct {
    PyObject_HEAD
    size_t number_of_reads;
    struct NanoInfo *nano_infos;
    size_t current_pos;
    PyObject *nano_stats;
} NanoStatsIterator;

static void NanoStatsIterator_dealloc(NanoStatsIterator *self) {
    Py_DECREF(self->nano_stats);
    Py_TYPE(self)->tp_free(self);
}

static PyTypeObject NanoStatsIterator_Type;

static PyObject *
NanoStatsIterator_FromNanoStats(NanoStats *nano_stats)
{
    NanoStatsIterator *self = PyObject_New(NanoStatsIterator, &NanoStatsIterator_Type);
    if (self == NULL) {
        return PyErr_NoMemory();
    }
    self->nano_infos = nano_stats->nano_infos;
    self->number_of_reads = nano_stats->number_of_reads;
    self->current_pos = 0;
    Py_INCREF(nano_stats);
    self->nano_stats = (PyObject *)nano_stats;
    return (PyObject *)self;
}

static PyObject *
NanoStatsIterator__iter__(NanoStatsIterator *self)
{
    Py_INCREF(self);
    return (PyObject *)self;
} 

static PyObject *
NanoStatsIterator__next__(NanoStatsIterator *self) {
    size_t current_pos = self->current_pos;
    if (current_pos == self->number_of_reads) {
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }
    NanoporeReadInfo *info = PyObject_New(NanoporeReadInfo, &NanoporeReadInfo_Type);
    if (info == NULL) {
        return PyErr_NoMemory();
    }
    memcpy(&info->info, self->nano_infos + current_pos, sizeof(struct NanoInfo));
    self->current_pos = current_pos + 1;
    return (PyObject *)info;
}

static PyTypeObject NanoStatsIterator_Type = {
    .tp_name = "_qc.NanoStatsIterator",
    .tp_basicsize = sizeof(NanoStatsIterator),
    .tp_dealloc = (destructor)NanoStatsIterator_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = (iternextfunc)NanoStatsIterator__iter__,
    .tp_iternext = (iternextfunc)NanoStatsIterator__next__,
};


static PyObject *
NanoStats__new__(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
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
NanoInfo_from_header(const uint8_t *header, size_t header_length, struct NanoInfo *info) 
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
        const uint8_t *field_end = memchr(field_value, ' ', end_ptr - field_value);
        if (field_end == NULL) {
            field_end = end_ptr;
        }
        cursor = field_end + 1;
        switch(field_name_length) {
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
        struct NanoInfo *tmp = PyMem_Realloc(self->nano_infos, new_size * sizeof(struct NanoInfo));
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

    if (meta->channel !=-1) {
        /* Already parsed from BAM */
        info->channel_id = meta->channel;
        info->duration = meta->duration;
        info->start_time = meta->start_time;
    }
    else if (NanoInfo_from_header(meta->record_start + 1, meta->name_length, info) != 0) {
        PyObject *header_obj = PyUnicode_DecodeASCII(
            (const char *)meta->record_start + 1, meta->name_length, NULL);
        if (header_obj == NULL) {
            return -1;
        }
        self->skipped = true;
        self->skipped_reason = PyUnicode_FromFormat(
            "Can not parse header: %R",
            header_obj);
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
"    A FastqRecordView object.\n"
);

#define NanoStats_add_read_method METH_O 

static PyObject *
NanoStats_add_read(NanoStats *self, FastqRecordView *read) 
{
    if (!FastqRecordView_CheckExact(read)) {
        PyErr_Format(PyExc_TypeError, 
                     "read should be a FastqRecordView object, got %s", 
                     Py_TYPE(read)->tp_name);
        return NULL;
    }
    if (NanoStats_add_meta(self, &read->meta) !=0) {
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
"    A FastqRecordArrayView object.\n"
);

#define NanoStats_add_record_array_method METH_O

static PyObject * 
NanoStats_add_record_array(
    NanoStats *self, FastqRecordArrayView *record_array) 
{
    if (!FastqRecordArrayView_CheckExact(record_array)) {
        PyErr_Format(PyExc_TypeError, 
                     "record_array should be a FastqRecordArrayView object, got %s", 
                     Py_TYPE(record_array)->tp_name);
        return NULL;
    }
    if (self->skipped) {
        Py_RETURN_NONE;
    }
    Py_ssize_t number_of_records = Py_SIZE(record_array);
    struct FastqMeta *records = record_array->records;
    for (Py_ssize_t i=0; i < number_of_records; i++) {
        if (NanoStats_add_meta(self, records + i) !=0) {
           return NULL;
        }
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(NanoStats_nano_info_iterator__doc__,
"nano_info_iterator($self, /)\n"
"--\n"
"\n"
"Return an iterator of NanoporeReadInfo objects. \n"
);

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
    {"skipped_reason", T_OBJECT, offsetof(NanoStats, skipped_reason),
     READONLY, "What the reason is for skipping the module if skipped." 
               "Set to None if not skipped."},
    {"minimum_time", T_LONG, offsetof(NanoStats, min_time), READONLY,
     "The earliest timepoint found in the headers",},
    {"maximum_time", T_LONG, offsetof(NanoStats, max_time), READONLY,
     "The latest timepoint found in the headers"},
    {NULL},
};

static PyTypeObject NanoStats_Type = {
    .tp_name = "_qc.NanoStats",
    .tp_basicsize = sizeof(NanoStats),
    .tp_dealloc = (destructor)NanoStats_dealloc,
    .tp_new = (newfunc)NanoStats__new__, 
    .tp_methods = NanoStats_methods,
    .tp_members = NanoStats_members,
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
    if (PythonArray == NULL) {
        return NULL;
    }
    if (python_module_add_type(m, &FastqParser_Type) != 0) {
        return NULL;
    }  
    if (python_module_add_type(m, &BamParser_Type) != 0) {
        return NULL;
    }  
    if (python_module_add_type(m, &FastqRecordView_Type) != 0) {
        return NULL; 
    }
    if (python_module_add_type(m, &FastqRecordArrayView_Type) != 0) {
        return NULL;
    }
    if (python_module_add_type(m, &QCMetrics_Type) != 0) {
        return NULL;
    }
    if (python_module_add_type(m, &Adapteruint64_type) != 0) {
        return NULL;
    }
    if (python_module_add_type(m, &PerTileQuality_Type) != 0) {
        return NULL;
    }
    if (python_module_add_type(m, &SequenceDuplication_Type) != 0) {
        return NULL;
    }
    if (python_module_add_type(m, &DedupEstimator_Type) != 0) {
        return NULL;
    }
    if (python_module_add_type(m, &NanoporeReadInfo_Type) != 0) {
        return NULL;
    }
    if (python_module_add_type(m, &NanoStats_Type) != 0) {
        return NULL;
    }
    if (python_module_add_type(m, &NanoStatsIterator_Type) != 0) {
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
    PyModule_AddIntMacro(m, DEFAULT_MAX_UNIQUE_FRAGMENTS);
    PyModule_AddIntMacro(m, DEFAULT_DEDUP_HASH_TABLE_SIZE_BITS);
    PyModule_AddIntMacro(m, DEFAULT_FRAGMENT_LENGTH);
    PyModule_AddIntMacro(m, DEFAULT_UNIQUE_SAMPLE_EVERY);
    return m;
}
