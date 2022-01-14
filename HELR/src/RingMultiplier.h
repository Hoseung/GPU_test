/*
 * Copyright (c) by CryptoLab inc.
 * This program is licensed under a
 * Creative Commons Attribution-NonCommercial 3.0 Unported License.
 * You should have received a copy of the license along with this
 * work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
 */
#ifndef HEAAN_RINGMULTIPLIER_H_
#define HEAAN_RINGMULTIPLIER_H_

#include <cstdint>
#include <vector>
#include <NTL/ZZ.h>


//#include "cu_ring_mult.h"

using namespace std;
//using namespace NTL;

class RingMultiplier {
public:

	long logN;
	long N;

	uint64_t* pVec;
	uint64_t* prVec;
	long* pTwok;
	uint64_t* pInvVec;
	uint64_t** scaledRootPows;
	uint64_t** scaledRootInvPows;
	uint64_t* scaledNInv;
	_ntl_general_rem_one_struct** red_ss_array;
	NTL::mulmod_precon_t** coeffpinv_array;
	NTL::ZZ* pProd;
	NTL::ZZ* pProdh;
	NTL::ZZ** pHat;
	uint64_t** pHatInvModp;

	RingMultiplier(long logN, long logQ);

	bool primeTest(uint64_t p);

	void NTT(uint64_t* a, long index);
	void INTT(uint64_t* a, long index);

	uint64_t* toNTT(NTL::ZZ* x, long np);

	void addNTTAndEqual(uint64_t* ra, uint64_t* rb, long np);

	void reconstruct(NTL::ZZ* x, uint64_t* rx, long np, NTL::ZZ& mod);

	void mult(NTL::ZZ* x, NTL::ZZ* a, NTL::ZZ* b, long np, NTL::ZZ& mod);

	void multNTT(NTL::ZZ* x, NTL::ZZ* a, uint64_t* rb, long np, NTL::ZZ& mod);

	void multDNTT(NTL::ZZ* x, uint64_t* ra, uint64_t* rb, long np, NTL::ZZ& mod);

	void multAndEqual(NTL::ZZ* a, NTL::ZZ* b, long np, NTL::ZZ& mod);

	void multNTTAndEqual(NTL::ZZ* a, uint64_t* rb, long np, NTL::ZZ& mod);

	void square(NTL::ZZ* x, NTL::ZZ* a, long np, NTL::ZZ& mod);

	void squareNTT(NTL::ZZ* x, uint64_t* ra, long np, NTL::ZZ& mod);

	void squareAndEqual(NTL::ZZ* a, long np, NTL::ZZ& mod);

	void mulMod(uint64_t& r, uint64_t a, uint64_t b, uint64_t p);

	void mulModBarrett(uint64_t& r, uint64_t a, uint64_t b, uint64_t p, uint64_t pr, long twok);

	uint64_t invMod(uint64_t x, uint64_t p);

	uint64_t powMod(uint64_t x, uint64_t y, uint64_t p);

	uint64_t inv(uint64_t x);

	uint64_t pow(uint64_t x, uint64_t y);

	uint32_t bitReverse(uint32_t x);

	void findPrimeFactors(vector<uint64_t> &s, uint64_t number);

	uint64_t findPrimitiveRoot(uint64_t m);

	uint64_t findMthRootOfUnity(uint64_t M, uint64_t p);


	// cuda function

	uint64_t* cuda_host_mult(NTL::ZZ* x, NTL::ZZ* a, NTL::ZZ* b, long np, NTL::ZZ& mod);

};

#endif /* RINGMULTIPLIER_H_ */
