#include "SkySphere.cuh"

#include "../../Vendor/stb_image/stb_image.h"

SkySphere::SkySphere(const char* filepath)
{
    stbi_set_flip_vertically_on_load(1);

    this -> data = stbi_loadf(filepath, &image_width, &image_height, &image_channels, 0);
    this -> isEnabled = true;

    if (data == nullptr)
    {
        fprintf(stderr, "FATAL: Image %s failed to load properly.", filepath);
        exit(EXIT_FAILURE);
    }
}

SkySphere::~SkySphere()
{
    stbi_image_free(data);
}

void SkySphere::prepareData()
{
    unsigned int dataSize = image_width * image_height * image_channels * sizeof(float);
    
    cudaMalloc(&deviceData, dataSize);
    cudaMemcpy(deviceData, data, dataSize, cudaMemcpyHostToDevice);
}

void SkySphere::freeData()
{
    cudaFree(deviceData);
}

void SkySphere::changeTexture(const char* filepath)
{
    stbi_set_flip_vertically_on_load(1);

    this -> data = stbi_loadf(filepath, &image_width, &image_height, &image_channels, 0);
    this -> isEnabled = true;

    if (data == nullptr)
    {
        fprintf(stderr, "FATAL: Image %s failed to load properly.", filepath);
        exit(EXIT_FAILURE);
    }
}

__device__ glm::vec3 SkySphere::getSkyColor(Ray* ray)
{
    if (isDark)
        return glm::vec3(0.0f);
    
    glm::vec3 rayDirection = ray -> getDirection();

    double u = 0.5 + atan2(rayDirection.z, rayDirection.x) / 6.28318;
    double v = 0.5 + asin(rayDirection.y) / 3.14159;

    double xout = glm::mix(0, image_width, u);
    double yout = glm::mix(0, image_height, v);

    unsigned int image_index = ((image_width * yout) + xout) * image_channels;

    // Prevent accessing illegal memory
    if (image_index >= image_width * image_height *image_channels)
        return glm::vec3(0.0f, 0.0f, 0.0f);

    // Don't forget gamma correction!!
    float r = glm::pow(deviceData[image_index], 1.0f / 2.2f);
    float g = glm::pow(deviceData[image_index + 1], 1.0f / 2.2f);
    float b = glm::pow(deviceData[image_index + 2], 1.0f / 2.2f);

    return glm::vec3(r,g,b);
}
