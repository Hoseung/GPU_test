#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <NTL/ZZ.h>
#include "cu_icrt.h"
#include <iostream>

#define MAX_SIZE 64
#define NP_MAX 64

__constant__  long c_n[1];
__constant__  long c_np[1];


__device__ void add_big_type( uint64_t* a,uint64_t* b)
{
    uint64_t carry = 0;
    for(int i =0;i<MAX_SIZE;i++)
    {
        a[i] = a[i] + b[i] + carry;
        carry = (a[i] < b[i]);
    }
}

__device__ void mult_big_type(uint64_t* result, uint64_t* a,long s)
{
    uint64_t carry = 0;
    uint64_t tmp =0;
    for(int i =0;i<MAX_SIZE;i++)
    {
        tmp = a[i]*s;
        tmp += carry;

        carry = __umul64hi(a[i],s) + (tmp < carry);

        result[i] = tmp;
    }
    
}

__device__ long reconst_inner(uint64_t rxi,uint64_t tt,uint64_t p,unsigned long ttpinv )
{
    uint64_t qq = __umul64hi(rxi,ttpinv);
    uint64_t rr = rxi * tt - qq * p;
    long rrl = long(rr);
    long pl = long(p);
    return rrl -pl >=0? rrl-pl : rrl;
}

__global__ void cu_reconst(uint64_t* result,uint64_t* pHatnp, uint64_t* rx,uint64_t* pVec,uint64_t* pHatInvModpnp,uint64_t* coeffpinv_arraynp)
{
    __shared__ uint64_t acc[64*MAX_SIZE];
    uint64_t rxi = rx[blockIdx.x + (threadIdx.x *c_n[0])];
    long s = reconst_inner(rxi,pHatInvModpnp[threadIdx.x],pVec[threadIdx.x],coeffpinv_arraynp[threadIdx.x]);

    mult_big_type(acc + MAX_SIZE*threadIdx.x, pHatnp + MAX_SIZE*threadIdx.x,s);
    
    __syncthreads();

    for(int i = NP_MAX>>1 ; i >0; i>>=1)
    {
      if(threadIdx.x < i && threadIdx.x+i < blockDim.x)
      {
        add_big_type(acc + MAX_SIZE*threadIdx.x,acc + MAX_SIZE*(threadIdx.x+i));
      }
      __syncthreads();   
    }
    __syncthreads();
    if(threadIdx.x ==0)
    {
      for(int i =0;i<MAX_SIZE;i++)
      {
          result[i + MAX_SIZE*blockIdx.x] = acc[i];
          //if(blockIdx.x ==0) printf("result %d %lu\n",i, acc[i]);
      }
      __syncthreads();
    }

}


cuda_reconstruct::cuda_reconstruct(long _n,long _np)
{
    n = _n;
    np = _np;
    pHatnp_splited = new uint64_t[MAX_SIZE * np];
    result = new uint64_t[MAX_SIZE * n];


    cudaMalloc(&c_pVec,np*sizeof(uint64_t));
    cudaMalloc(&c_pHatInvModpnp,np*sizeof(uint64_t));
    cudaMalloc(&c_coeffpinv_arraynp,np*sizeof(uint64_t));
    cudaMalloc(&c_result,MAX_SIZE*n*sizeof(uint64_t));
    cudaMalloc(&c_pHatnp,MAX_SIZE*np*sizeof(uint64_t));

    cudaMalloc(&c_rx,n*np*sizeof(uint64_t));

    cudaMemcpyToSymbol(c_n,&n,sizeof(long),0,cudaMemcpyHostToDevice);

}

cuda_reconstruct::~cuda_reconstruct()
{


    cudaFree(c_pVec);
    cudaFree(c_pHatInvModpnp);
    cudaFree(c_coeffpinv_arraynp);
    cudaFree(c_result);
    cudaFree(c_pHatnp);
    cudaFree(c_rx);

}

void cuda_reconstruct::copy_param(NTL::ZZ* pHatnp, uint64_t* pVec, uint64_t* pHatInvModpnp, unsigned long* coeffpinv_arraynp)
{
    cudaMemset(&c_pHatnp,0,np*MAX_SIZE*sizeof(uint64_t));
    for(long j =0;j<np;j++)
    {
        NTL::ZZ a_data = pHatnp[j];
        long size = a_data.size();
        if(size == 0 ) size = 1;

        uint64_t* tmp = new uint64_t[size]{0,};
        for(long i =0;i < size ; i++)
        {
            tmp[i] = NTL::conv<uint64_t>(a_data);
            a_data >>= 64;
        }
        cudaMemcpy(c_pHatnp + j*MAX_SIZE, tmp, (size) *sizeof(uint64_t),cudaMemcpyHostToDevice);
    }

    cudaMemcpy(c_pVec,pVec,np*sizeof(uint64_t),cudaMemcpyHostToDevice);
    cudaMemcpy(c_pHatInvModpnp, pHatInvModpnp, np*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(c_coeffpinv_arraynp, coeffpinv_arraynp, np*sizeof(uint64_t), cudaMemcpyHostToDevice);

}

void cuda_reconstruct::return_data(NTL::ZZ* x,NTL::ZZ& pProdnp,NTL::ZZ& pProdhnp,NTL::ZZ& mod)
{
    cudaMemcpy(result,c_result,MAX_SIZE*n*sizeof(uint64_t),cudaMemcpyDeviceToHost);

    for(int i =0;i<n;i++)
    {


        for(int j =0;j<MAX_SIZE;j++)
        {
            if (result[MAX_SIZE*i+j]==0) break;

            NTL::ZZ tmp = NTL::conv<NTL::ZZ>(result[MAX_SIZE*i+j]);
            tmp = tmp << MAX_SIZE*j;
            x[i] += tmp;
            //printf("tt: %lu\n",result[MAX_SIZE*i+j]);

        }
        
        NTL::QuickRem(x[i], pProdnp);
		if (x[i] > pProdhnp) x[i] -= pProdnp;
		NTL::QuickRem(x[i], mod);
        
    }

}
void cuda_reconstruct::reconst(uint64_t* rx)
{  

    cudaMemcpy(c_rx,rx,np*n*sizeof(uint64_t),cudaMemcpyHostToDevice);


    dim3 grids(n,1,1);
    dim3 threads(np,1,1);
    cu_reconst<<<grids,threads>>>(c_result,c_pHatnp, c_rx,c_pVec,c_pHatInvModpnp, c_coeffpinv_arraynp);

    //printf("done %lu %lu\n",n,np);
}

void cuda_reconstruct::icrt_to_host(NTL::ZZ* x, NTL::ZZ* pProd, NTL::ZZ* pProdh, NTL::ZZ& mod)
{
    return_data(x,pProd[np-1],pProdh[np-1],mod);
}
