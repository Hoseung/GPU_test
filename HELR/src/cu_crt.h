
#include <NTL/ZZ.h>


class cuda_rem
{
private:
    long* c_p;
    uint64_t* c_tlb;
    uint64_t* c_data;
    
    bool* c_neg_check;

    long np;
    long n;
    uint64_t* c_result;
    _ntl_general_rem_one_struct* red_ss;

public:
    cuda_rem(long _n,long _np);

    ~cuda_rem();

    void cuda_tbl_build(uint64_t* pVec);

    //void cuda_ZZ_to_uint(NTL::ZZ* data,long pi);
    void cuda_ZZ_to_uint(NTL::ZZ* data);

    void cuda_rem_calc();

    void result_to_host(uint64_t* result);

};