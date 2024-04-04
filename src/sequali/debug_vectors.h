#include "immintrin.h"

__attribute__((__target__("avx2")))
static inline void print_si256(__m256i vec, char *name) {
    static uint8_t vec_store[32];
    _mm256_storeu_si256((__m256i *)vec_store, vec);
    printf("%s:\n", name);
    for (size_t i=0; i<32; i++){
        printf("%02X, ", vec_store[i]);
        if ((i & 15) == 15) {
            printf("\n");
        }
    }
}

__attribute__((__target__("avx2")))
static inline void print_si256_as_string(__m256i vec, char *name) {
    static char vec_store[33];
    _mm256_storeu_si256((__m256i *)vec_store, vec);
    vec_store[32] = 0;
    printf("%s: %s\n", name, vec_store);
}