

#include "RingMultiplier.h"
#include "cu_ring_mult.h"
#include "cu_crt.h"
#include "cu_ntt.h"
#include "cu_icrt.h"
//#include <time.h>
#include <chrono>

chrono::high_resolution_clock::time_point g1, g2;
#define START() g1 = chrono::high_resolution_clock::now();
#define END() g2 = chrono::high_resolution_clock::now();
#define SUM(t) t += (double)chrono::duration_cast<chrono::nanoseconds >(g2-g1).count() ;
#define PRINTTIME(msg) std::cout << msg << " time = " << (double)chrono::duration_cast<std::chrono::nanoseconds >(g2 - g1).count() / 1000 << " microseconds" << std::endl;




uint64_t* RingMultiplier::cuda_host_mult(NTL::ZZ* x, NTL::ZZ* a, NTL::ZZ* b, long np, NTL::ZZ& mod)
{
    double full_time = 0;
    uint64_t* ra = new uint64_t[np << logN]();
    uint64_t* rb = new uint64_t[np << logN]();
    uint64_t* rx = new uint64_t[np << logN]();
    
    chrono::high_resolution_clock::time_point start1, end1;

    start1 = chrono::high_resolution_clock::now();
    cuda_NTT_handler ntt_handle(np,logN);

    START()
    ntt_handle.ntt_memAlloc(np,logN);
    

    ntt_handle.ntt_param_cpy(pVec,prVec,pInvVec,pTwok,scaledRootPows,scaledRootInvPows,scaledNInv);
    END() PRINTTIME("\n ntt_mem_cpy")

    cuda_rem rem_handler(N,np);
    
    rem_handler.cuda_tbl_build(pVec);
    cudaDeviceSynchronize();
    
    START()
    rem_handler.cuda_ZZ_to_uint(a);
    END() PRINTTIME("CRT mem_cpy time")

    cudaDeviceSynchronize();
    START()
    rem_handler.cuda_rem_calc();
    END() PRINTTIME("CRT run time")
    SUM(full_time)

    rem_handler.result_to_host(ra);
    
    
    rem_handler.cuda_ZZ_to_uint(b);
    cudaDeviceSynchronize();
    START()
    rem_handler.cuda_rem_calc();
    END()
    SUM(full_time)

    rem_handler.result_to_host(rb);
    
    
    
    // After CRT-RNS; proceed GPU allocation;
    START()
    ntt_handle.ntt_poly_cpy(ra,rb);
    END() PRINTTIME("NTT mem time")
    START()
    ntt_handle.cuda_NTT();
    END() PRINTTIME("NTT run time")
    SUM(full_time)

    START()
    ntt_handle.mulModBarrett_cuda();
    END() PRINTTIME("mulMod run time")
    SUM(full_time)

    START()
    ntt_handle.cuda_INTT();
    END()
    SUM(full_time)

    ntt_handle.ntt_to_host(rx);
    //reconstruct(x,rx,np,mod);
    
    //std::cout << "value good: " << x[0] << std::endl;
    cuda_reconstruct recon_handler(N,np);
    recon_handler.copy_param(pHat[np-1],pVec,pHatInvModp[np-1],coeffpinv_array[np-1]);

    
    
    START()
    recon_handler.reconst(rx);
    END() PRINTTIME("ICRT run time")
    SUM(full_time)

    START()
    recon_handler.icrt_to_host(x,pProd,pProdh,mod);
    END() PRINTTIME("ICRT mem time")

    end1 = chrono::high_resolution_clock::now();
    
    //std::cout << "value test: " << x[0] << std::endl;

    std::cout << "GPU run time: " << full_time/1000 << endl;
    std::cout << "GPU full time = " << (double)chrono::duration_cast<std::chrono::nanoseconds >(end1 - start1).count() / 1000 << " microseconds" << std::endl;

    return rx;
    delete[] ra;
    delete[] rb;
    delete[] rx;
}
