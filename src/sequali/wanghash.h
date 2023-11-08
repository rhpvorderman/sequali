#include <stdint.h>

// Thomas Wang's integer hash functions and inverse
// See https://naml.us/post/inverse-of-a-hash-function/ 

/* MSVC version 2022 makes some critical inline errors for the inverse hash
   function */
#ifdef _MSC_VER
#define noinline __declspec(noinline)
#else
#define noinline
#endif

static noinline uint64_t wanghash64(uint64_t key) {
	key = (~key) + (key << 21); // key = (key << 21) - key - 1;
	key = key ^ (key >> 24);
	key = (key + (key << 3)) + (key << 8); // key * 265
	key = key ^ (key >> 14);
	key = (key + (key << 2)) + (key << 4); // key * 21
	key = key ^ (key >> 28);
	key = key + (key << 31);
	return key;
}


static noinline uint64_t wanghash64_inverse(uint64_t key) {
	uint64_t tmp;

	// Invert key = key + (key << 31)
	tmp = key-(key<<31);
	key = key-(tmp<<31);

	// Invert key = key ^ (key >> 28)
	tmp = key^key>>28;
	key = key^tmp>>28;

	// Invert key *= 21
	key *= 14933078535860113213u;

	// Invert key = key ^ (key >> 14)
	tmp = key^key>>14;
	tmp = key^tmp>>14;
	tmp = key^tmp>>14;
	key = key^tmp>>14;

	// Invert key *= 265
	key *= 15244667743933553977u;

	// Invert key = key ^ (key >> 24)
	tmp = key^key>>24;
	key = key^tmp>>24;

	// Invert key = (~key) + (key << 21)
	tmp = ~key;
	tmp = ~(key-(tmp<<21));
	tmp = ~(key-(tmp<<21));
	key = ~(key-(tmp<<21));

	return key;
}
