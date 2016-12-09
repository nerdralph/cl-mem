#include "config.h"

/*
 * OpenCL memory benchmarks 
 */

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
    uint block;
    uint blockend = (MEMSIZE/sizeof(uint))/WAVES;
    uint rep = REPS;

    while (rep--)
    {
         __global uint *p = (__global uint *)
                            (buffer + wave * (MEMSIZE/WAVES));
        for (block = 0; block < blockend; block += 256)
        {
            *(p + block + tid) = tid;
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
    uint blockend = (MEMSIZE/sizeof(uint))/WAVES;
    uint rep = REPS;
    uint sum = 0;

    while (rep--)
    {
         __global uint *p = (__global uint *)
                            (buffer + wave * (MEMSIZE/WAVES));
        for (block = 0; block < blockend; block += 256)
        {
            sum += *(p + block + tid);
        }
        //*(p + block + tid) = sum;
    }
    // write sum to keep compiler from optimizing away the whole kernel
    *(__global uint *)(buffer + wave * (MEMSIZE/WAVES)) = sum;
}

