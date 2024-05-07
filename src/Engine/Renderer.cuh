#ifndef RENDERER_H
#define RENDERER_H

#include "../Common.h"

#include "Ray.cuh"
#include "Objects/Cube.cuh"
#include "Objects/Sphere.cuh"
#include "Objects/SkySphere.cuh"

typedef struct _DisplayInfo
{
    int width;
    int height;
    
    float* pixels;
    float* accumulatedPixels;

    int frameIndex;

} DisplayInfo;

typedef struct _CameraInfo
{
    glm::vec3 cameraPosition;
    glm::vec3 cameraDirection;

} CameraInfo;

void Render(DisplayInfo& displayInfo, CameraInfo* cameraInfo, std::vector<Sphere>* spheres, std::vector<Cube>* cubes, SkySphere* sky, glm::mat4* viewMatrixInverse, glm::mat4* projectionMatrixInverse, uint32_t& seed);

#endif // RENDERER_H