#include "config.h"

/*
 * OpenCL memory benchmarks 
 */

/* each group of threads aka WAVE uses a different section of memory */
#define DTYPE ulong
#define BLOCKEND ((MEMSIZE/sizeof(DTYPE))/WAVES)
#define INITPTR __global DTYPE *p = (__global DTYPE *)\
                                    (buffer + wave * (MEMSIZE/WAVES))

/*
 * sequential write 1GB
 * should be launched with local worksize 256
 * WAVES should be a power of 2
 */
__kernel
void test0(__global char *buffer)
{
    uint tid = get_global_id(0)%256;
    uint wave = get_global_id(0)>>8;  // 256 threads per wave
    DTYPE data = get_global_id(0) << 16;
    uint block;
    uint rep = REPS;

    while (rep--)
    {
        INITPTR;
        for (block = 0; block < BLOCKEND; block += 256)
        {
            *(p + block + tid) = data;
        }
    }
}

/*
 * sequential read 1GB
 * should be launched with local worksize 256
 * WAVES should be a power of 2
 */
__kernel
void test1(__global char *buffer)
{
    uint tid = get_global_id(0)%256;
    uint wave = get_global_id(0)>>8;  // 256 threads per wave
    uint block;
    uint rep = REPS;
    DTYPE sum = 0;

    while (rep--)
    {
        INITPTR;
        for (block = 0; block < BLOCKEND; block += 256)
        {
            sum += *(p + block + tid);
        }
    }
    // write sum to keep compiler from optimizing away the whole kernel
    *(__global DTYPE *)(buffer + wave * (MEMSIZE/WAVES)) = sum;
}

/*
 * sequential copy 1GB/2
 * should be launched with local worksize 256
 * WAVES should be a power of 2
 */
__kernel
void test2(__global char *buffer)
{
    uint tid = get_global_id(0)%256;
    uint wave = get_global_id(0)>>8;  // 256 threads per wave
    uint block;
    uint rep = REPS;

    while (rep--)
    {
        INITPTR;
        for (block = 0; block*2 < BLOCKEND; block += 256)
        {
            // 1st half of block is src, 2nd half is dst
            *(p + BLOCKEND/2 + block + tid) = *(p + block + tid);
        }
    }
}

