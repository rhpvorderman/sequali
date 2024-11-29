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

#include "compiler_defs.h"

#include <string.h>

struct Entry {
    Py_ssize_t score;
    Py_ssize_t query_matches;
};

static int8_t
get_smith_waterman_matches_default(
    const uint8_t *restrict target, size_t target_length,
    const uint8_t *restrict query, size_t query_length, int8_t match_score,
    int8_t mismatch_penalty, int8_t deletion_penalty, int8_t insertion_penalty)
{
    Py_ssize_t highest_score = 0;
    Py_ssize_t most_matches = 0;
    struct Entry prev_column[32];
    struct Entry new_column[32];
    memset(prev_column, 0, 32 * sizeof(struct Entry));
    memset(new_column, 0, 32 * sizeof(struct Entry));
    for (size_t i = 0; i < target_length; i++) {
        uint8_t target_char = target[i];
        for (size_t j = 1; j < query_length + 1; j++) {
            uint8_t query_char = query[j - 1];
            struct Entry prev_entry = prev_column[j - 1];
            Py_ssize_t linear_score;
            Py_ssize_t linear_matches;
            if (target_char == query_char) {
                linear_score = prev_entry.score + match_score;
                linear_matches = prev_entry.query_matches + 1;
            }
            else {
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
            }
            else if (insertion_score >= deletion_score) {
                /* When an insertion happens in the query in theory we can
                   match all query characeters still. So deduct one as a penalty. */
                score = insertion_score;
                matches = prev_ins_entry.query_matches - 1;
            }
            else {
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
            }
            else if (score > highest_score) {
                highest_score = score;
                most_matches = matches;
            }
        }
        memcpy(prev_column, new_column, sizeof(prev_column));
    }
    return most_matches;
}

static int8_t (*get_smith_waterman_matches)(
    const uint8_t *restrict target, size_t target_length,
    const uint8_t *restrict query, size_t query_length, int8_t match_score,
    int8_t mismatch_penalty, int8_t deletion_penalty,
    int8_t insertion_penalty) = get_smith_waterman_matches_default;

#if COMPILER_HAS_TARGETED_DISPATCH && BUILD_IS_X86_64

/**
 * @brief Shift everything one byte down. Similar to _mm256_bslli_epi128, but
 * does shift over the 128-bit lanes.
 */
__attribute__((__target__("avx2"))) static inline __m256i
_mm256_move_one_down(__m256i vec)
{
    /* This moves the vector one down, but the entry at position 15 will be
       missing, as the shift does not go beyond the 128-bit lanes. */
    __m256i shifted_vec = _mm256_bsrli_epi128(vec, 1);
    /* Rotate 64 bit integers: 1, 2, 3, 0. */
    __m256i rotated_vec = _mm256_permute4x64_epi64(vec, 0x39);  // 0b00111001
    /* shift 7 bytes up. This brings the right byte to position 15. */
    __m256i shifted_rotated_vec = _mm256_slli_epi64(rotated_vec, 56);
    __m256i masked_vec =
        _mm256_and_si256(_mm256_setr_epi64x(0, -1, 0, 0), shifted_rotated_vec);
    return _mm256_or_si256(shifted_vec, masked_vec);
}

__attribute__((__target__("avx2")))
/**
 * Get the smith waterman matches by an avx2 algorithm. This goes over
 * diagonals rather than columns, as the diagonals are independent.
 */
static int8_t
get_smith_waterman_matches_avx2(const uint8_t *restrict target,
                                size_t target_length,
                                const uint8_t *restrict query,
                                size_t query_length, int8_t match_score,
                                int8_t mismatch_penalty, int8_t deletion_penalty,
                                int8_t insertion_penalty)
{
    /* Since the algorithm goes over reversed diagonals, it needs to be padded
       with  the vector length - 1 on both sides. So vectors can be loaded
       immediately rather than have a complex initialization. */
    uint8_t *padded_target = PyMem_Calloc(target_length + 62, 1);
    if (padded_target == NULL) {
        return -1;
    }
    memcpy(padded_target + 31, target, target_length);

    /* Also the query needs to be fitted inside a vector.*/
    uint8_t padded_query_store[32];
    uint8_t *padded_query = padded_query_store;
    /* Pacters than target */
    memset(padded_query, 0xff, sizeof(padded_query_store));
    for (size_t i = 0; i < query_length; i++) {
        size_t index = 31 - i;
        padded_query[index] = query[i];
    }
    __m256i query_vec = _mm256_lddqu_si256((__m256i *)padded_query);

    __m256i max_matches = _mm256_setzero_si256();
    __m256i max_score = _mm256_setzero_si256();
    __m256i prev_diagonal_score = _mm256_setzero_si256();
    __m256i prev_prev_diagonal_score = _mm256_setzero_si256();
    __m256i prev_diagonal_matches = _mm256_setzero_si256();
    __m256i prev_prev_diagonal_matches = _mm256_setzero_si256();
    __m256i match_score_vec = _mm256_set1_epi8(match_score);
    __m256i mismatch_penalty_vec = _mm256_set1_epi8(mismatch_penalty);
    __m256i deletion_penalty_vec = _mm256_set1_epi8(deletion_penalty);
    __m256i insertion_penalty_vec = _mm256_set1_epi8(insertion_penalty);

    size_t run_length = target_length + query_length;
    for (size_t i = 0; i < run_length; i++) {
        __m256i target_vec = _mm256_lddqu_si256((__m256i *)(padded_target + i));

        /* Normally the previous diagonals need to be loaded from arrays, but
           since the query length is limited at 31, the previous diagonals can
           be stored in registers instead. */
        __m256i prev_linear_score = prev_prev_diagonal_score;
        __m256i prev_linear_matches = prev_prev_diagonal_matches;
        __m256i prev_insertion_score = prev_diagonal_score;
        __m256i prev_insertion_matches = prev_diagonal_matches;
        /* Rather than loading from a +1 offset from memory, instead use
           vector instructions to achieve the same effect. This is much faster
           but only works since the query is at most 31 bp. */
        __m256i prev_deletion_score = _mm256_move_one_down(prev_diagonal_score);
        __m256i prev_deletion_matches =
            _mm256_move_one_down(prev_diagonal_matches);

        __m256i query_equals_target = _mm256_cmpeq_epi8(target_vec, query_vec);
        __m256i linear_score_if_equals =
            _mm256_add_epi8(prev_linear_score, match_score_vec);
        __m256i linear_score_if_not_equals =
            _mm256_add_epi8(prev_linear_score, mismatch_penalty_vec);
        __m256i linear_matches_if_equals =
            _mm256_add_epi8(prev_linear_matches, _mm256_set1_epi8(1));
        __m256i linear_score =
            _mm256_blendv_epi8(linear_score_if_not_equals,
                               linear_score_if_equals, query_equals_target);
        __m256i linear_matches = _mm256_blendv_epi8(
            prev_linear_matches, linear_matches_if_equals, query_equals_target);
        __m256i deletion_score =
            _mm256_add_epi8(prev_deletion_score, deletion_penalty_vec);
        __m256i deletion_matches = prev_deletion_matches;
        __m256i insertion_score =
            _mm256_add_epi8(prev_insertion_score, insertion_penalty_vec);
        __m256i insertion_matches =
            _mm256_sub_epi8(prev_insertion_matches, _mm256_set1_epi8(1));

        /* For calculating the scores, simple max instructions suffice. But the
           matches vector needs to line up with the score vector so some
           comparators and blend instructions are needed for it. */
        __m256i insertion_greater_than_linear =
            _mm256_cmpgt_epi8(insertion_score, linear_score);
        __m256i deletion_greater_than_linear =
            _mm256_cmpgt_epi8(deletion_score, linear_score);
        __m256i deletion_greater_than_insertion =
            _mm256_cmpgt_epi8(deletion_score, insertion_score);
        __m256i deletion_greatest = _mm256_and_si256(
            deletion_greater_than_insertion, deletion_greater_than_linear);
        __m256i insertion_greatest = _mm256_andnot_si256(
            deletion_greatest, insertion_greater_than_linear);
        __m256i indels_greatest =
            _mm256_or_si256(insertion_greatest, deletion_greatest);
        __m256i indel_score = _mm256_max_epi8(insertion_score, deletion_score);
        __m256i indel_matches = _mm256_blendv_epi8(
            insertion_matches, deletion_matches, deletion_greater_than_insertion);
        __m256i scores = _mm256_max_epi8(linear_score, indel_score);
        __m256i matches =
            _mm256_blendv_epi8(linear_matches, indel_matches, indels_greatest);
        __m256i zero_greater_than_scores =
            _mm256_cmpgt_epi8(_mm256_setzero_si256(), scores);
        matches = _mm256_blendv_epi8(matches, _mm256_setzero_si256(),
                                     zero_greater_than_scores);
        scores = _mm256_max_epi8(scores, _mm256_setzero_si256());

        __m256i equal_score = _mm256_cmpeq_epi8(scores, max_score);
        __m256i greater_score = _mm256_cmpgt_epi8(scores, max_score);
        __m256i tmp_max_matches = _mm256_max_epi8(matches, max_matches);
        max_score = _mm256_max_epi8(scores, max_score);
        max_matches =
            _mm256_blendv_epi8(max_matches, tmp_max_matches, equal_score);
        max_matches = _mm256_blendv_epi8(max_matches, matches, greater_score);
        prev_prev_diagonal_score = prev_deletion_score;
        prev_prev_diagonal_matches = prev_deletion_matches;
        prev_diagonal_score = scores;
        prev_diagonal_matches = matches;
    }
    uint8_t max_score_store[32];
    uint8_t max_matches_store[32];
    int8_t highest_score = 0;
    int8_t best_matches = 0;
    _mm256_storeu_si256((__m256i *)max_score_store, max_score);
    _mm256_storeu_si256((__m256i *)max_matches_store, max_matches);
    for (size_t i = 0; i < 32; i++) {
        int8_t score = max_score_store[i];
        int8_t matches = max_matches_store[i];
        if (score == highest_score && matches > best_matches) {
            best_matches = matches;
        }
        else if (score > highest_score) {
            highest_score = score;
            best_matches = matches;
        }
    }
    PyMem_Free(padded_target);
    return best_matches;
}

__attribute__((constructor)) static void
get_smith_waterman_matches_dispatch(void)
{
    if (__builtin_cpu_supports("avx2")) {
        get_smith_waterman_matches = get_smith_waterman_matches_avx2;
    }
    else {
        get_smith_waterman_matches = get_smith_waterman_matches_default;
    }
}
#endif

PyDoc_STRVAR(
    sequence_identity__doc__,
    "Calculate sequence identity based on a smith-waterman matrix. Only keep\n"
    "two columns in memory as no walk-back is needed.\n"
    "Identity is given as (query_length - errors / query_length).\n");

#define sequence_identity_method METH_VARARGS | METH_KEYWORDS

static PyObject *
sequence_identity(PyObject *module, PyObject *args, PyObject *kwargs)
{
    static char *format = "UU|nnnn:identify_sequence";
    static char *kwnames[] = {"target",
                              "query",
                              "match_score",
                              "mismatch_penalty",
                              "deletion_penalty",
                              "insertion_penalty",
                              NULL};
    PyObject *target_obj = NULL;
    PyObject *query_obj = NULL;
    Py_ssize_t match_score = 1;
    Py_ssize_t mismatch_penalty = -1;
    Py_ssize_t deletion_penalty = -1;
    Py_ssize_t insertion_penalty = -1;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, kwnames, &target_obj,
                                     &query_obj, &match_score, &mismatch_penalty,
                                     &deletion_penalty, &insertion_penalty)) {
        return NULL;
    }
    Py_ssize_t target_length = PyUnicode_GetLength(target_obj);
    Py_ssize_t query_length = PyUnicode_GetLength(query_obj);
    Py_ssize_t target_utf8_length = 0;
    Py_ssize_t query_utf8_length = 0;
    const uint8_t *target = (const uint8_t *)PyUnicode_AsUTF8AndSize(
        target_obj, &target_utf8_length);
    const uint8_t *query =
        (const uint8_t *)PyUnicode_AsUTF8AndSize(query_obj, &query_utf8_length);
    if (target_length != target_utf8_length) {
        PyErr_Format(PyExc_ValueError,
                     "Only ascii strings are allowed. Got %R", target_obj);
        return NULL;
    }
    if (query_length != query_utf8_length) {
        PyErr_Format(PyExc_ValueError,
                     "Only ascii strings are allowed. Got %R", target_obj);
        return NULL;
    }

    if (query_length > 31) {
        PyErr_Format(
            PyExc_ValueError,
            "Only query with lengths less than 32 are supported. Got %zd",
            query_length);
        return NULL;
    }
    Py_ssize_t most_matches = get_smith_waterman_matches(
        target, target_length, query, query_length, match_score,
        mismatch_penalty, deletion_penalty, insertion_penalty);
    if (most_matches < 0) {
        return PyErr_NoMemory();
    }
    double identity = (double)most_matches / (double)query_length;
    return PyFloat_FromDouble(identity);
}

static PyMethodDef _seqident_methods[] = {
    {"sequence_identity", (PyCFunction)sequence_identity,
     sequence_identity_method, sequence_identity__doc__},
    {NULL},
};

static PyModuleDef_Slot _seqident_module_slots[] = {
    {0, NULL},
};

static struct PyModuleDef _seqident_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_seqident",
    .m_doc = NULL,
    .m_size = 0,
    .m_methods = _seqident_methods,
    .m_slots = _seqident_module_slots,
};

PyMODINIT_FUNC
PyInit__seqident(void)
{
    return PyModuleDef_Init(&_seqident_module);
}
