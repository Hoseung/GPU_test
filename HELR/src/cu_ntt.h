#include <cooperative_groups.h>
#include <cuda.h> 
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

class cuda_NTT_handler
{
private:
    
    long coeff_size;
    long crt_coeff_size;
    long np;
    long logN;
    long N;

    // param for NTT
    uint64_t* c_pVec;
    uint64_t* c_prVec;
    uint64_t* c_pInvVec;
    long* c_pTwok;
    uint64_t* c_scaledRootPows;
    uint64_t* c_scaledRootInvPows;
    uint64_t* c_scaledNInv;

    // poly for NTT
    uint64_t* c_ra;
    uint64_t* c_rb;
    uint64_t* c_rx;
    cudaStream_t stream_mem;

public:

cuda_NTT_handler(long np_,long logN_);

~cuda_NTT_handler();

void ntt_memAlloc(long np_,long logN_);

void ntt_param_cpy(uint64_t* pVec, uint64_t* prVec, uint64_t* pInvVec, long* pTwok, uint64_t** scaledRootPows,uint64_t** scaledRootInvPows, uint64_t* scaledNInv);

void ntt_poly_cpy(uint64_t* ra,uint64_t* rb);

void ntt_to_host(uint64_t* x);

void cuda_NTT();

void cuda_INTT();

void cuda_NTT_run(uint64_t* c_poly_ring,int thread_max = 8);

void cuda_INTT_run(uint64_t* c_poly_ring,int thread_max = 8);


void mulModBarrett_cuda(int thread_max = 8);
};