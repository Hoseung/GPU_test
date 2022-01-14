#include <iostream>
#include <cuda.h> 
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <NTL/ZZ.h>
#include "cu_crt.h"

#define CUDA_BIT_PER_LONG 64
#define CUDA_REMAIN_SIZE 128
// remaineder size is 128+4
#define MAX_COEFF_SIZE 64
struct ll_type{
  uint64_t hi,lo;
};

__device__  void divmod128by64(const uint64_t u1, const uint64_t u0, uint64_t v, uint64_t& q, uint64_t& r);
__device__  uint64_t modmul(uint64_t a, uint64_t b, uint64_t mod);
__device__ void ll_mul(ll_type& x, uint64_t a, uint64_t b);
__device__ void acuumulate(ll_type& x, ll_type& y);

__global__ void struct_build(uint64_t* _tlb, long* _p)
{  
    long p = _p[threadIdx.x];
    uint64_t* tlb = _tlb + threadIdx.x*(CUDA_REMAIN_SIZE+4);


    long t =1;
    for(long j=0;j< CUDA_BIT_PER_LONG;j++)
    {
        t+=t;
        if(t >=p) t-=p;
    }
    __syncthreads();
    long t1=1;
    long t2=t;
    tlb[0]=1;
    for(long j=1;j<CUDA_REMAIN_SIZE;j++)
    {
        t1 = modmul(t1,t,p);
        tlb[j] = t1;
    }
    for(long j=CUDA_REMAIN_SIZE+1;j<CUDA_REMAIN_SIZE+3;j++)
    {
        t1 = modmul(t1,t2,p);
        tlb[j] = t1;
    }

}

__global__ void rem(uint64_t* result, uint64_t* _a_data,uint64_t* tbl,long* p,bool* neg_check)
{
  uint64_t* a_data = _a_data + MAX_COEFF_SIZE * blockIdx.x;
  __shared__ ll_type acc[MAX_COEFF_SIZE];//acc.hi = 0; acc.lo=0;
  acc[threadIdx.x].hi = 0;
  acc[threadIdx.x].lo = 0;
  ll_mul(acc[threadIdx.x],a_data[threadIdx.x],tbl[threadIdx.x +(CUDA_REMAIN_SIZE+4)*blockIdx.y]);
  // if(blockIdx.y == 1)
  // {
  //   printf("%d: %lu %lu  \n",threadIdx.x,a_data[threadIdx.x],tbl[threadIdx.x +(CUDA_REMAIN_SIZE+4)*blockIdx.y] ) ;
  // }
  __syncthreads();
  for(int i = MAX_COEFF_SIZE>>1 ; i >0; i>>=1)
  {
    if(threadIdx.x < i)
    {
      //printf("%d %d: %lu %lu  \n",threadIdx.x, threadIdx.x +i,acc[threadIdx.x].lo,acc[threadIdx.x +i].lo ) ;
      acuumulate(acc[threadIdx.x],acc[threadIdx.x +i]);
      //printf("%lu %lu  %lu %lu || ",i,threadIdx.x,acc[threadIdx.x].hi,acc[threadIdx.x].lo);     
    }
    __syncthreads();   
  }
  __syncthreads();
  uint64_t q;
  if(threadIdx.x ==0)
  {
    divmod128by64(acc[0].hi,acc[0].lo,p[blockIdx.y],q,result[gridDim.x*blockIdx.y+ blockIdx.x]);
    if(neg_check[blockIdx.x] == true)
    {
      result[gridDim.x*blockIdx.y+ blockIdx.x] = p[blockIdx.y] -result[gridDim.x*blockIdx.y+ blockIdx.x];
    }
  }
  //if(threadIdx.x ==0)printf("%d %d %lu\n",blockIdx.x,blockIdx.y,result[gridDim.x*blockIdx.y+ blockIdx.x]);

}

__device__  void divmod128by64( uint64_t u1, const uint64_t u0, uint64_t v, uint64_t& q, uint64_t& r) 
{
    if (u1 > v) u1 = u1%v;

    // apply when u1 is smaller than v
    const uint64_t b = 1ll << 32;
    uint64_t un1, un0, vn1, vn0, q1, q0, un32, un21, un10, rhat, left, right;
    size_t s;
  
    s = __clzll(v); //cuda count leading zeros
    v <<= s;
    vn1 = v >> 32;
    vn0 = v & 0xffffffff;
  
    if (s > 0)
      {
        un32 = (u1 << s) | (u0 >> (64 - s));
        un10 = u0 << s;
      }
    else
      {
        un32 = u1;
        un10 = u0;
      }
  
    un1 = un10 >> 32;
    un0 = un10 & 0xffffffff;
  
    q1 = un32 / vn1;
    rhat = un32 % vn1;
  
    left = q1 * vn0;
    right = (rhat << 32) + un1;
   again1:
    if ((q1 >= b) || (left > right))
      {
        --q1;
        rhat += vn1;
        if (rhat < b)
          {
            left -= vn0;
            right = (rhat << 32) | un1;
            goto again1;
          }
      }
  
    un21 = (un32 << 32) + (un1 - (q1 * v));
  
    q0 = un21 / vn1;
    rhat = un21 % vn1;
  
    left = q0 * vn0;
    right = (rhat << 32) | un0;
   again2:
    if ((q0 >= b) || (left > right))
      {
        --q0;
        rhat += vn1;
        if (rhat < b)
          {
            left -= vn0;
            right = (rhat << 32) | un0;
            goto again2;
          }
      }
  
    r = ((un21 << 32) + (un0 - (q0 * v))) >> s;
    q = (q1 << 32) | q0;
}
   
//modulo multiplication using division.
__device__  uint64_t modmul(uint64_t a, uint64_t b, uint64_t mod)
{
    uint64_t result, w_hi(0), w_lo(0), q(0);

    w_lo = a * b;
    w_hi =  __umul64hi(a, b);
    divmod128by64(w_hi, w_lo, mod, q, result);
    return result;
}

__device__ void ll_mul(ll_type& x, uint64_t a, uint64_t b)
{
    x.hi = __umul64hi(a,b);
    x.lo = a*b;
    // if (x.hi <hi)
    // {
    //   printf("overflow\n");
    // }

}
__device__ void acuumulate(ll_type& x, ll_type& y)
{
  x.lo = x.lo +  y.lo;
  x.hi = x.hi + y.hi + (x.lo < y.lo);
  
}

__global__ void print_result(uint64_t* result,long n, long np)
{
    // for(int i =0;i<n;i++)
    // {
    //     printf("\n");
    //     for(int j =0; j<np;j++)
    //     {
    //         printf("%lu ",result[n*j+i]);
            
    //     }
    //     printf("\n");
    // }
}

cuda_rem::cuda_rem(long _n,long _np)
{
    np = _np;
    n = _n;
    cudaMalloc(&c_p, np* sizeof(long));
    cudaMalloc(&c_tlb,np*(CUDA_REMAIN_SIZE+4)*sizeof(uint64_t));
    cudaMalloc(&c_data,n*MAX_COEFF_SIZE*sizeof(uint64_t));
    cudaMemset(&c_data,0,n*MAX_COEFF_SIZE*sizeof(uint64_t));
    cudaMalloc(&c_result,n*np*sizeof(uint64_t));


    cudaMalloc(&c_neg_check,n*sizeof(bool));
}
cuda_rem::~cuda_rem()
{
    cudaFree(c_p);
    cudaFree(c_tlb);
    cudaFree(c_data);
    cudaFree(c_result);
    cudaFree(c_neg_check);
}

void cuda_rem::cuda_tbl_build(uint64_t* pVec)
{
    //red_ss = _ntl_general_rem_one_struct_build(pVec[0]);
    cudaMemcpy(c_p,pVec,np*sizeof(long),cudaMemcpyHostToDevice);

    dim3 grids(1,1,1);
    dim3 threads(np,1,1);
    struct_build<<<grids,threads>>>(c_tlb,c_p);
}

void cuda_rem::cuda_ZZ_to_uint(NTL::ZZ* data)
{
    cudaMemset(&c_data,0,n*MAX_COEFF_SIZE*sizeof(uint64_t));
    // initiailize value to zero
    
    // uint64_t answer;
    // answer = _ntl_general_rem_one_struct_apply(a_data[0].rep, pi, red_ss);
    // std::cout <<"answer:  "<< answer << std::endl;
    
    // temporary solution for minus value
    bool* neg_check = new bool[n];
    //
    for(long j =0;j<n;j++)
    {
      NTL::ZZ a_data = data[j];
      long size = a_data.size();
      if(size == 0 ) size = 1;
      
      if (a_data < 0)
      {
        a_data = -a_data;
        neg_check[j] = true;
      }
      else{
        neg_check[j] = false;
      }

      uint64_t* tmp = new uint64_t[size]{0,};
      for(long i =0;i < size ; i++)
      {
        tmp[i] = NTL::conv<uint64_t>(a_data);
        a_data >>= 64;
      }
      cudaMemcpy( c_data+j*MAX_COEFF_SIZE, tmp,(size)*sizeof(uint64_t),cudaMemcpyHostToDevice);
    }
    cudaMemcpy( c_neg_check, neg_check,(n)*sizeof(bool),cudaMemcpyHostToDevice);        
}
void cuda_rem::cuda_rem_calc()
{
    dim3 grids(n,np,1);
    dim3 threads(MAX_COEFF_SIZE,1,1);
    rem<<<grids,threads>>>(c_result, c_data, c_tlb, c_p,c_neg_check);

}
void cuda_rem::result_to_host(uint64_t* result)
{
    cudaMemcpy(result, c_result,np*n*sizeof(uint64_t),cudaMemcpyDeviceToHost);
}