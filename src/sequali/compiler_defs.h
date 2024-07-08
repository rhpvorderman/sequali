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
#define GCC_AT_LEAST(major, minor) 0
#endif

#ifdef __clang__
#ifdef __has_attribute
#define CLANG_COMPILER_HAS(attribute) __has_attribute(attribute)
#endif
#ifdef __has_builtin
#define CLANG_COMPILER_HAS_BUILTIN(function) __has_builtin(function)
#endif
#endif
#ifndef CLANG_COMPILER_HAS
#define CLANG_COMPILER_HAS(attribute) 0
#endif
#ifndef CLANG_COMPILER_HAS_BUILTIN
#define CLANG_COMPILER_HAS_BUILTIN(function) 0
#endif

#define COMPILER_HAS_TARGETED_DISPATCH                                             \
    (GCC_AT_LEAST(4, 8) || (CLANG_COMPILER_HAS(__target__) &&                      \
                            CLANG_COMPILER_HAS_BUILTIN(__builtin_cpu_supports)) && \
                               CLANG_COMPILER_HAS(constructor))

#define COMPILER_HAS_OPTIMIZE \
    (GCC_AT_LEAST(4, 4) || CLANG_COMPILER_HAS(optimize))

#if defined(__x86_64__) || defined(_M_X64)
#define BUILD_IS_X86_64 1
#include "immintrin.h"
#else
#define BUILD_IS_X86_64 0
#endif
