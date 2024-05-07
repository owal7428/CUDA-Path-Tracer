#include "util.hpp"

// Taken from https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
__host__ __device__ uint32_t pcg_hash(uint32_t input)
{
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__host__ __device__ float randomFloat(uint32_t& seed)
{
    seed = pcg_hash(seed);
    return (float) seed / UINT32_MAX;
}

// Implementation of Box-Muller transform https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform#Basic_form
__host__ __device__ float randomFloatInNormal(uint32_t& seed)
{
    float R = sqrt(-2 * log(randomFloat(seed)));
    float theta = 2 * glm::pi<float>() * randomFloat(seed);
    return R * cos(theta);
}

// Taken from https://math.stackexchange.com/questions/1585975/how-to-generate-random-points-on-a-sphere
__host__ __device__ glm::vec3 randomVec3InHemisphere(uint32_t& seed)
{
    float x = randomFloatInNormal(seed);
    float y = randomFloatInNormal(seed);
    float z = randomFloatInNormal(seed);

    return glm::normalize(glm::vec3(x,y,z));
}

__host__ __device__ glm::vec3 randomVec3(uint32_t& seed)
{
    for (int i = 0; i < 100; i++)
    {
        float x = randomFloat(seed) * 2.0f - 1.0f;
        float y = randomFloat(seed) * 2.0f - 1.0f;
        float z = randomFloat(seed) * 2.0f - 1.0f;

        glm::vec3 point = glm::vec3(x, y, z);

        float magnitudeSquared = glm::dot(point, point);

        if (magnitudeSquared <= 1)
        {
            return point / magnitudeSquared;
        }
    }

    return glm::vec3(0.0f);
}

__device__ glm::vec3 randomVec3Device(uint32_t seed)
{
    for (int i = 0; i < 100; i++)
    {
        float x = randomFloat(seed) * 2.0f - 1.0f;
        float y = randomFloat(seed) * 2.0f - 1.0f;
        float z = randomFloat(seed) * 2.0f - 1.0f;

        glm::vec3 point = glm::vec3(x, y, z);

        float magnitudeSquared = glm::dot(point, point);

        if (magnitudeSquared <= 1)
        {
            return point / magnitudeSquared;
        }
    }

    return glm::vec3(0.0f);
}

__device__ float randomFloatDevice(uint32_t seed)
{
    seed = pcg_hash(seed);
    return (float) seed / UINT32_MAX;
}
