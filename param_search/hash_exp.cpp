// Example program
#include <iostream>
#include <string>
#include <unordered_map>
typedef unsigned int uint32;

/* 32 bit FNV-1  non-zero initial basis */
#define FNV1_32_INIT ((uint32)0x811c9dc5)

/* Constant prime value used for an FNV1 hash */
#define FNV_32_PRIME ((uint32)0x01000193)




static uint32
fnv1_32_buf(void *buf, size_t len, uint32 hval) {
	unsigned char *bp = (unsigned char *) buf;	/* start of buffer */
	unsigned char *be = bp + len;		/* beyond end of buffer */

	/*
	 * FNV-1 hash each octet in the buffer
	 */
	while (bp < be) {
		/* multiply by the 32 bit FNV magic prime mod 2^32 */
#if defined(NO_FNV_GCC_OPTIMIZATION)
		hval *= FNV_32_PRIME;
#else
		hval += (hval << 1) + (hval << 4) + (hval << 7) + (hval << 8) + (hval << 24);
#endif

		/* xor the bottom with the current octet */
		hval ^= (uint32) * bp++;
	}

	/* return our new hash value */
	return hval;
}



int main() {
	std::unordered_map<int, int> buckets;
	for (int j = 1; j < 2000; ++j){
		for (int i = 0; i < 99999; ++i) {
			long int intbuf = (long int) i;
			void	   *buf = NULL;
			buf = &intbuf;
			size_t len = sizeof(intbuf);
			uint32 hval = fnv1_32_buf(buf, len, FNV1_32_INIT);
			int dist_key = hval % j;
			buckets[dist_key] += 1;
			// std::cout << i << ' ' << hval << ' ' << hval %  20 << std::endl;
		}
		if (buckets.size() != j){
			std::cout << buckets.size() << "fail" << std::endl;
		}

	}
	std::cout << "success";

	return 0;

}
