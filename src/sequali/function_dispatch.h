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

// Macros also used in htslib, very useful.
#if defined __GNUC__
#define GCC_AT_LEAST(major, minor) \
    (__GNUC__ > (major) || (__GNUC__ == (major) && __GNUC_MINOR__ >= (minor)))
#else 
# define GCC_AT_LEAST(major, minor) 0
#endif

#ifdef __clang__
#ifdef __has_attribute
#define CLANG_COMPILER_HAS(attribute) __has_attribute(attribute)
#endif 
#endif
#ifndef CLANG_COMPILER_HAS 
#define CLANG_COMPILER_HAS(attribute) 0
#endif 

#define COMPILER_HAS_TARGET (GCC_AT_LEAST(4, 8) || CLANG_COMPILER_HAS(__target__))
#define COMPILER_HAS_OPTIMIZE (GCC_AT_LEAST(4,4) || CLANG_COMPILER_HAS(optimize))

#if defined(__x86_64__) || defined(_M_X64)
#define BUILD_IS_X86_64 1
#include "immintrin.h"
#else 
#define BUILD_IS_X86_64 0
#endif

#include <stdint.h>
#include <string.h>
#include <stddef.h>

static void 
decode_bam_sequence_default(uint8_t *dest, const uint8_t *encoded_sequence, size_t length)  {
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
    for (size_t i=0; i < length_2; i++) {
        memcpy(dest + i*2, code2base + ((size_t)encoded_sequence[i] * 2), 2);
    }
    if (length & 1) {
        uint8_t encoded = encoded_sequence[length_2] >> 4;
        dest[(length - 1)] = nuc_lookup[encoded];
    }
}

#if COMPILER_HAS_TARGET && BUILD_IS_X86_64
__attribute__((__target__("ssse3")))
static void 
decode_bam_sequence_ssse3(uint8_t *dest, const uint8_t *encoded_sequence, size_t length) 
{

    static const uint8_t *nuc_lookup = (uint8_t *)"=ACMGRSVTWYHKDBN";
    const uint8_t *dest_end_ptr = dest + length;
    uint8_t *dest_cursor = dest;
    const uint8_t *encoded_cursor = encoded_sequence;
    const uint8_t *dest_vec_end_ptr = dest_end_ptr - (2 * sizeof(__m128i));
    __m128i first_upper_shuffle = _mm_setr_epi8(
        0, -1, 1, -1, 2, -1, 3, -1, 4, -1, 5, -1, 6, -1, 7, -1);
    __m128i first_lower_shuffle = _mm_setr_epi8(
        -1, 0, -1, 1, -1, 2, -1, 3, -1, 4, -1, 5, -1, 6, -1, 7);
    __m128i second_upper_shuffle = _mm_setr_epi8(
        8, -1, 9, -1, 10, -1, 11, -1, 12, -1, 13, -1, 14, -1, 15, -1);
    __m128i second_lower_shuffle = _mm_setr_epi8(
        -1, 8, -1, 9, -1, 10, -1, 11, -1, 12, -1, 13, -1, 14, -1, 15);
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
        Bitwise and with 0b1111 (15) leads to:
                     |0A|0B|0C|0D|0E|0F|0G|0H|
        We can use the resulting 4-bit integers as indexes for the shuffle of 
        the nucleotide lookup. */
    while (dest_cursor < dest_vec_end_ptr) {
        __m128i encoded = _mm_lddqu_si128((__m128i *)encoded_cursor);

        __m128i first_upper = _mm_shuffle_epi8(encoded, first_upper_shuffle);
        __m128i first_lower = _mm_shuffle_epi8(encoded, first_lower_shuffle);
        __m128i shifted_first_upper = _mm_srli_epi64(first_upper, 4);
        __m128i first_merged = _mm_or_si128(shifted_first_upper, first_lower);
        __m128i first_indexes = _mm_and_si128(first_merged, _mm_set1_epi8(15));
        __m128i first_nucleotides = _mm_shuffle_epi8(nuc_lookup_vec, first_indexes);
        _mm_storeu_si128((__m128i *)dest_cursor, first_nucleotides);

        __m128i second_upper = _mm_shuffle_epi8(encoded, second_upper_shuffle);
        __m128i second_lower = _mm_shuffle_epi8(encoded, second_lower_shuffle);
        __m128i shifted_second_upper = _mm_srli_epi64(second_upper, 4);
        __m128i second_merged = _mm_or_si128(shifted_second_upper, second_lower);
        __m128i second_indexes = _mm_and_si128(second_merged, _mm_set1_epi8(15));
        __m128i second_nucleotides = _mm_shuffle_epi8(nuc_lookup_vec, second_indexes);
        _mm_storeu_si128((__m128i *)(dest_cursor + 16), second_nucleotides);

        encoded_cursor += sizeof(__m128i);
        dest_cursor += 2 * sizeof(__m128i);
    }
    decode_bam_sequence_default(dest_cursor, encoded_cursor, dest_end_ptr - dest_cursor);
}

static void (*decode_bam_sequence)(uint8_t *dest, const uint8_t *encoded_sequence, size_t length);

static void decode_bam_sequence_dispatch(uint8_t *dest, const uint8_t *encoded_sequence, size_t length) {
    if (__builtin_cpu_supports("ssse3")) {
        decode_bam_sequence = decode_bam_sequence_ssse3;
    }
    else {
        decode_bam_sequence = decode_bam_sequence_default;
    }
    decode_bam_sequence(dest, encoded_sequence, length);
}

static void (*decode_bam_sequence)(uint8_t *dest, const uint8_t *encoded_sequence, size_t length) = decode_bam_sequence_dispatch;

#else
static inline void decode_bam_sequence(uint8_t *dest, const uint8_t *encoded_sequence, size_t length) 
{
    decode_bam_sequence_default(dest, encoded_sequence, length);
}
#endif 

// Code is simple enough to be auto vectorized.
#if COMPILER_HAS_OPTIMIZE
__attribute__((optimize("O3")))
#endif
static void 
decode_bam_qualities(
    uint8_t *restrict dest,
    const uint8_t *restrict encoded_qualities,
    size_t length)
{
    for (size_t i=0; i<length; i++) {
        dest[i] = encoded_qualities[i] + 33;
    }
}


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
    /* Compiler properly recognizes the above as a byteswap and will simplify
       using the bswap instruction. */
    revcomp = ((revcomp & 0xF0F0F0F0F0F0F0F0ULL) >> 4) |
              ((revcomp & 0x0F0F0F0F0F0F0F0FULL) << 4);
    revcomp = ((revcomp & 0xCCCCCCCCCCCCCCCCULL) >> 2) |
              ((revcomp & 0x3333333333333333ULL) << 2);
    // If k < 32, the empty twobit slots will have ended up at the least
    // significant bits. Use a shift to move them to the highest bits again.
    return revcomp >> (64 - (k *2));
}

static int64_t sequence_to_canonical_kmer_default(uint8_t *sequence, uint64_t k) {
    uint64_t kmer = 0;
    size_t all_nucs = 0;
    int64_t i=0;
    int64_t vector_end = k - 4;
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
    for (i=i; i<(int64_t)k; i++) {
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

#if COMPILER_HAS_TARGET && BUILD_IS_X86_64
__attribute__((__target__("avx2")))
static int64_t sequence_to_canonical_kmer_avx2(uint8_t *sequence, uint64_t k) {
    /* By using a load mask, at most 3 extra bytes are loaded. Given that a 
       sequence in sequali always ends with \n+\n this should not trigger invalid 
       memory access.*/
   __m256i load_mask = _mm256_cmpgt_epi32(
        _mm256_add_epi32(
            _mm256_set1_epi32((k + 3) / 4),
            _mm256_setr_epi32(0, -1, -2, -3, -4, -5, -6, -7)
        ), 
        _mm256_setzero_si256()
    );
   __m256i seq_vec_raw = _mm256_maskload_epi32((int *)sequence, load_mask);
    /* Use only the last 3 bits to create indices from 0-15. A,C,G and T are 
        distinct in the last 3 bits. This will yield results for any 
        input. The non-ACGT check is performed at the end of the function.
    */
   __m256i indices_vec = _mm256_and_si256(_mm256_set1_epi8(7), seq_vec_raw);
   /* Use the shufb instruction to convert the 0-7 indices to corresponding 
      ACGT twobit representation. Everything non C, G, T will be 0, the same as 
      A. */
   __m256i twobit_vec = _mm256_shuffle_epi8(_mm256_setr_epi8(
        /*     A,  , C, T,  ,  , G */
            0, 0, 0, 1, 3, 0, 0, 2,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 3, 0, 0, 2,
            0, 0, 0, 0, 0, 0, 0, 0
        ), indices_vec
    );
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
        twobit_vec,
        _mm256_slli_epi16(twobit_vec, 2), 
        alternate_byte_select
    );
    __m256i twobit_shifted_vec = _mm256_blendv_epi8(
        first_shift,
        _mm256_slli_epi16(first_shift, 4), 
        alternate_word_select
    );
    /* Now the groups of four need to be bitwise ORred together. Due to the 
       way the data is prepared ADD and OR have the same effect. We can use
       _mm256_sad_epu8 instruction with a zero function to horizontally add 
       8-bit integers. Since this adds 8-bit integers in groups of 8, we use
       a mask to select only 4 bytes. We can then use a shift and a or to 
       get all resulting integers into one vector again.
    */
    __m256i four_select_mask = _mm256_set1_epi64x(0x00000000FFFFFFFF);
    __m256i upper_twobit = _mm256_sad_epu8(_mm256_and_si256(four_select_mask, twobit_shifted_vec), _mm256_setzero_si256());
    __m256i lower_twobit = _mm256_sad_epu8(_mm256_andnot_si256(four_select_mask, twobit_shifted_vec), _mm256_setzero_si256());
    __m256i combined_twobit = _mm256_or_si256(_mm256_bslli_epi128(upper_twobit, 1), lower_twobit);

    /* The following instructions arrange the 8 resulting twobit bytes in the 
       correct order to be extracted as a 64 bit integer. */
    __m256i packed_twobit = _mm256_shuffle_epi8(combined_twobit,_mm256_setr_epi8(
        8, 9, 0, 1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, 
        8, 9, 0, 1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1
    ));
    __m256i shuffled_twobit = _mm256_permutevar8x32_epi32(packed_twobit, _mm256_setr_epi32(4, 0, 7, 7, 7, 7, 7, 7));
    uint64_t kmer = _mm_cvtsi128_si64(_mm256_castsi256_si128(shuffled_twobit));
    kmer =  kmer >> (64 - (k *2));

    /* NON-ACGT CHECK*/

    /* In order to mask only k characters.
      Create an array with only k. Create an array with 0, -1, -2, -3 etc. 
      Add k array to descending array. If k=2 descending array will be,
      2, 1, 0, -1, etc. cmpgt with zero array results in, yes, yes, no, no  etc. 
      First two characters masked with k=2.
   */
   __m256i seq_mask = _mm256_cmpgt_epi8(
        _mm256_add_epi8(
            _mm256_set1_epi8(k),
            _mm256_setr_epi8(
                0, -1, -2, -3, -4, -5, -6, -7, 
                -8, -9, -10, -11, -12, -13, -14, -15, 
                -16, -17, -18, -19, -20, -21, -22, -23,
                -24, -25, -26, -27, -28, -29, -30, -31
            )
        ), 
        _mm256_setzero_si256()
    );
    /* Mask all characters not of interest as A to not false positively trigger
       the non-ACGT detection */
    __m256i seq_vec = _mm256_blendv_epi8(_mm256_set1_epi8('A'), seq_vec_raw, seq_mask);
    /* 32 is the ASCII lowercase bit. Use and not to make everything upper case. */
    __m256i seq_vec_upper = _mm256_andnot_si256(_mm256_set1_epi8(32), seq_vec);
    __m256i ACGT_vec = _mm256_or_si256(
        _mm256_or_si256(
            _mm256_cmpeq_epi8(seq_vec_upper, _mm256_set1_epi8('A')),
            _mm256_cmpeq_epi8(seq_vec_upper, _mm256_set1_epi8('C'))
        ), 
        _mm256_or_si256(
            _mm256_cmpeq_epi8(seq_vec_upper, _mm256_set1_epi8('G')),
            _mm256_cmpeq_epi8(seq_vec_upper, _mm256_set1_epi8('T'))

        )
    );
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

static int64_t (*sequence_to_canonical_kmer)(uint8_t *sequence, uint64_t k);

static int64_t sequence_to_canonical_kmer_dispatch(uint8_t *sequence, uint64_t k) 
{
    if (__builtin_cpu_supports("avx2")) {
        sequence_to_canonical_kmer = sequence_to_canonical_kmer_avx2;
    } 
    else {
        sequence_to_canonical_kmer = sequence_to_canonical_kmer_default;
    }
    return sequence_to_canonical_kmer(sequence, k);
}

static int64_t (*sequence_to_canonical_kmer)(
    uint8_t *sequence, uint64_t k) = sequence_to_canonical_kmer_dispatch;

#else 
static inline int64_t sequence_to_canonical_kmer(uint8_t *sequence, uint64_t k) {
    return sequence_to_canonical_kmer_default(sequence, k);
}
#endif