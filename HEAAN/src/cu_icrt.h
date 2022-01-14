#include <NTL/ZZ.h>

class cuda_reconstruct
{
private:
    uint64_t* pHatnp_splited;
    uint64_t* c_pHatnp;
    
    uint64_t* result;
    uint64_t* c_result;
    
    uint64_t* c_pVec;
    uint64_t* c_pHatInvModpnp;
    uint64_t* c_coeffpinv_arraynp;

    uint64_t* c_rx;
    

    long np,n;

public:
    cuda_reconstruct(long _n,long _np);
    ~cuda_reconstruct();
    void copy_param(NTL::ZZ* pHatnp, uint64_t* pVec, uint64_t* pHatInvModpnp, unsigned long* coeffpinv_arraynp);
    
    void return_data(NTL::ZZ* x,NTL::ZZ& pProdnp,NTL::ZZ& pProdhnp,NTL::ZZ& mod);
    void reconst(uint64_t* rx);
    void icrt_to_host(NTL::ZZ* x, NTL::ZZ* pProd, NTL::ZZ* pProdh, NTL::ZZ& mod);

};
