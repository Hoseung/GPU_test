
#include <iostream>
#include <cuda.h> 
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <NTL/ZZ.h>

#define CUDA_BIT_PER_LONG 64
#define CUDA_REMAIN_SIZE 128
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

__global__ void rem(uint64_t* result, uint64_t* _a_data,uint64_t* tbl,long* p)
{
  uint64_t* a_data = _a_data + MAX_COEFF_SIZE * blockIdx.x;
  __shared__ ll_type acc[MAX_COEFF_SIZE];//acc.hi = 0; acc.lo=0;


  // multi_thread
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
  divmod128by64(acc[0].hi,acc[0].lo,p[blockIdx.y],q,result[gridDim.x*blockIdx.y+ blockIdx.x]);
  if(threadIdx.x ==0)printf("%d %d %lu\n",blockIdx.x,blockIdx.y,result[gridDim.x*blockIdx.y+ blockIdx.x]);

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


class cuda_rem
{
private:
  long* c_p;
  uint64_t* c_tlb;
  uint64_t* c_data;
  uint64_t* c_result;

  long np;
  long n;
  _ntl_general_rem_one_struct* red_ss;

public:
  cuda_rem(long _n,long _np)
  {
    np = _np;
    n = _n;
    cudaMalloc(&c_p, np* sizeof(long));
    cudaMalloc(&c_tlb,np*(CUDA_REMAIN_SIZE+4)*sizeof(uint64_t));
    cudaMalloc(&c_data,n*MAX_COEFF_SIZE*sizeof(uint64_t));
    cudaMemset(&c_data,0,n*MAX_COEFF_SIZE*sizeof(uint64_t));
    cudaMalloc(&c_result,n*np*sizeof(uint64_t));
  }
  ~cuda_rem()
  {
    cudaFree(c_p);
    cudaFree(c_tlb);
    cudaFree(c_data);
  }

  void cuda_tbl_build(uint64_t* pVec)
  {
    red_ss = _ntl_general_rem_one_struct_build(pVec[0]);
    cudaMemcpy(c_p,pVec,np*sizeof(long),cudaMemcpyHostToDevice);

    dim3 grids(1,1,1);
    dim3 threads(np,1,1);
    struct_build<<<grids,threads>>>(c_tlb,c_p);
  }

  void cuda_ZZ_to_uint(NTL::ZZ* data,long index,long pi)
  {
    cudaMemset(&c_data,0,n*MAX_COEFF_SIZE*sizeof(uint64_t));
    // initiailize value to zero
    
    // uint64_t answer;
    // answer = _ntl_general_rem_one_struct_apply(a_data[0].rep, pi, red_ss);
    // std::cout <<"answer:  "<< answer << std::endl;
    
    
    for(long j =0;j<n;j++)
    {
      NTL::ZZ a_data = data[j];
      long size = a_data.size();
      if(size == 0 ) size = 1;

      uint64_t* tmp = new uint64_t[size]{0,};
      for(long i =0;i < size ; i++)
      {
        tmp[i] = NTL::conv<uint64_t>(a_data);
        a_data >>= 64;
      }
      cudaMemcpy( c_data+j*MAX_COEFF_SIZE, tmp,(size)*sizeof(uint64_t),cudaMemcpyHostToDevice);    
    }
  }
  void cuda_rem_calc()
  {
    dim3 grids(n,np,1);
    dim3 threads(MAX_COEFF_SIZE,1,1);
    rem<<<grids,threads>>>(c_result, c_data, c_tlb, c_p);
  }

};

int main()
{ 
    NTL::ZZ x[3];
    x[0] = NTL::conv<NTL::ZZ>("186855674353235446702462431270275192118543202011383815453076227294398878802040931592943506188089466495788740306343361080037460982929706433578073630180159919627773001769496934857110134289712078065110380365400353159253500098954035335523967732576287172236869503747050559852870672677598350494771253796760054694602324045105615832197691971065958495997657317741112890296038010319275976782861097661251361866543055418424747011497787985654630824335148664536383633481202572971369288268247561479465479720924087320848239834924836548897954333329103926480527831731546603021445752231102440597247527472053025304796587475120409570926993110093980023614014967867475235165021446764690038621741263301618295653801180644");
    x[1] = NTL::conv<NTL::ZZ>("47920939178644856853504724407266516764930253324680521743487474585486832759716894834113853608032442076444777665057904579000239981963857272245650932069766257310152342754357640606511886960579688063505470876921724586344338281782555947552438718279945110941377231309383388696039930548154903847810382545946847578345467163761642227960476811196796272691045696768037981120454715908058896877448120701779645431750118901755835889332968986010209478764868133612201986932612133088065048162622499157994340861605978013343175122642768549983098655316784656782274601350104699442019700942907339870777514618979797911711834243912086678054445747498214446498957445842325655533927360068718495756395629370276303313716246723");
    x[2] = NTL::conv<NTL::ZZ>("24812587921683634475957367807622899089868199726085646241574097154041775693622301664376778749257516870419142727844094045047789187179492074715351244055001503151526038787826518871410075067522438022979608938566420400025404098774268743503306202447662371743228344854448727551085572960642699415217559671274027151331126983978863839884718669855826805145314246850024206976170475309521091405770292560352053151892544014314135351834154401303163419260410885548660642851866921497296149188921454293834560225887822530406066360817348036428688434005326503014044958054642130721368577395874102689702339394006805651660956976372653669236537865822091217611500337034562383096605083324208377564982141658139498445414066580");
  //std::cout << x << std::endl;
    //long* p; cudaMallocManaged(&p,sizeof(long));
    uint64_t p[40] = {576460752308273153,576460752315482113,576460752319021057,576460752319414273,576460752321642497,576460752325705729,576460752328327169,576460752329113601,576460752329506817,576460752329900033,576460752331210753,576460752337502209,576460752340123649,576460752342876161,576460752347201537,576460752347332609,576460752352837633,576460752354017281,576460752355065857,576460752355459073,576460752358604801,576460752364240897,576460752368435201,576460752371187713,576460752373547009,576460752374333441,576460752376692737,576460752378003457,576460752378396673,576460752380755969,576460752381411329,576460752386129921,576460752395173889,576460752395960321,576460752396091393,576460752396484609,576460752399106049,576460752405135361,576460752405921793,576460752409722881};
    //uint64_t* tlb = new uint64_t[CUDA_REMAIN_SIZE+4];
    // uint64_t* a_data = new uint64_t[37];
    // a_data[0]=576460752409722880;
    // a_data[1]=576460752409722880;
    // a_data[2]=123412341234;
    //cudaMallocManaged(&tlb,(CUDA_REMAIN_SIZE+4)*sizeof(long));

    // long* c_p;
    // uint64_t* c_tlb;
    // cudaMalloc(&c_p,sizeof(long));
    // cudaMalloc(&c_tlb,(CUDA_REMAIN_SIZE+4)*sizeof(uint64_t));
  

    // cudaMemcpy(c_p,p,sizeof(long),cudaMemcpyHostToDevice);
    // cudaMemcpy(c_tlb,tlb,(CUDA_REMAIN_SIZE+4)*sizeof(uint64_t),cudaMemcpyHostToDevice);

    cuda_rem rem_handler(3,3);
    rem_handler.cuda_tbl_build(p);
    cudaDeviceSynchronize();
    rem_handler.cuda_ZZ_to_uint(x,0,p[0]);
    cudaDeviceSynchronize();
    rem_handler.cuda_rem_calc();
    cudaDeviceSynchronize();
    //cudaMemcpy(tlb,c_tlb,(CUDA_REMAIN_SIZE+4)*sizeof(uint64_t),cudaMemcpyDeviceToHost);
    printf("\n hello world");
    // for(long i =0; i<(CUDA_REMAIN_SIZE+4);i++ )
    // std::cout << tlb[i]<< " ";
}