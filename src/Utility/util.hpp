#ifndef UTIL_H
#define UTIL_H

#include "../Common.h"

// Random generators

__host__ __device__ float randomFloat(uint32_t& seed);
__host__ __device__ glm::vec3 randomVec3(uint32_t& seed);
__device__ glm::vec3 randomVec3Device(uint32_t seed);
__device__ float randomFloatDevice(uint32_t seed);

// Cuda error check

// Taken from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#endif // UTIL_H
