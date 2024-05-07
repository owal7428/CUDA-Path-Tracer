#ifndef COMMON_H
#define COMMON_H

// C includes
#include <stdio.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// C++ / CUDA includes
#if defined(__cplusplus) || defined(__CUDACC__)

#include <cuda_runtime.h>

#include <vector>

#include "Vendor/glm/glm/glm.hpp"
#include "Vendor/glm/glm/gtc/matrix_transform.hpp"
#include "Vendor/glm/glm/gtc/quaternion.hpp"
#include "Vendor/glm/glm/gtx/quaternion.hpp"
#include "Vendor/glm/glm/gtc/type_ptr.hpp"

#endif // __cplusplus

#endif // COMMON_H
