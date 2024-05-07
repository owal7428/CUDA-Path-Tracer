#include "Renderer.cuh"

#include "../Utility/util.hpp"

#include "../Vendor/stb_image/stb_image.h"

__device__ glm::vec3 skyColor(Ray* ray, SkySphere* sky)
{
    if (sky->getIsEnabled())
        return sky -> getSkyColor(ray);
    
    float coeff = glm::dot(ray -> getDirection(), glm::vec3(0.0f, 1.0f, 0.0f));

    if (coeff <= 0)
        return glm::vec3(0.5f, 0.49f, 0.43f);

    return glm::vec3(0.9f) - coeff * glm::vec3(0.53f, 0.2f, 0.0f);
}

__global__ void pixelColor(float* pixels, float* accumulatedPixels, int frameIndex, int width, int height, Sphere* spheres, int numSpheres, Cube* cubes, int numCubes, SkySphere* sky,
                            glm::vec3 cameraPosition, glm::mat4 viewMatrixInverse, glm::mat4 projectionMatrixInverse, uint32_t seed)
{
    // Screen space coordinates (0 to width/height - 1)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int index = ((width * y) + x) * 4;

    // Avoid writing to illegal memory address
    if (index >= width * height * 4)
        return;

    // Normalized device coordinates (-1 to 1)
    float u = (2 * (float) x / width) - 1;
    float v = (2 * (float) y / height) - 1;

    glm::vec3 origin = cameraPosition;
    glm::vec2 coords = glm::vec2(u, v);

    Ray pixelRay = Ray(origin, coords, &viewMatrixInverse, &projectionMatrixInverse);

    glm::vec3 light = glm::vec3(0.0f);
    glm::vec3 rayColor = glm::vec3(1.0f);

    // 10 bounces
    int bounces = 10;
    for (int i = 0; i < bounces; i++)
    {
        RayHitInfo rayInfo = {1e100, glm::vec3(0.0f)};
        Sphere* nearestSphere = nullptr;
        Cube* nearestCube = nullptr;

        // Find nearest object collision
        for (int j = 0; j < numSpheres; j++)
        {
            RayHitInfo check = spheres[j].checkHit(&pixelRay);

            // If collision is behind camera or further than closest collision then continue
            if (check.t <= 0 || check.t >= rayInfo.t )
                continue;
            
            rayInfo = check;
            nearestSphere = &spheres[j];
        }

        for (int j = 0; j < numCubes; j++)
        {
            RayHitInfo check = cubes[j].checkHit(&pixelRay);

            // If collision is behind camera or further than closest collision then continue
            if (check.t <= 0 || check.t >= rayInfo.t )
                continue;
            
            rayInfo = check;
            nearestCube = &cubes[j];
        }

        if (nearestSphere == nullptr && nearestCube == nullptr)
        {
            light += skyColor(&pixelRay, sky) * rayColor;
            break;
        }

        // If cube was closer than any sphere
        if (nearestCube != nullptr)
        {
            // Calculate bounced ray

            glm::vec3 point = pixelRay.getOrigin() + (float) rayInfo.t * pixelRay.getDirection();
            glm::vec3 normal = rayInfo.normal;

            glm::vec3 refPoint = point + 0.00001f * normal;

            pixelRay.setOrigin(refPoint);

            // Cosine weighted direction
            glm::vec3 diffuseDirection = glm::normalize(normal + randomVec3Device(seed + (uint32_t) (1645*x + y + 652*i + 8792*frameIndex)));
            glm::vec3 specularDirection = glm::reflect(pixelRay.getDirection(), normal);

            bool isSpecular = nearestCube->getMetallic() >= randomFloatDevice(seed + (uint32_t) (1645*x + y + 652*i + 8792*frameIndex));
            
            glm::vec3 newDirection = glm::mix(diffuseDirection, specularDirection, (1.0f - nearestCube->getRoughness()) * isSpecular);
            
            pixelRay.setDirection(newDirection);

            // Calculate lighting

            glm::vec3 emittedLight = nearestCube -> getEmission();
            light += emittedLight * rayColor;

            rayColor *= glm::mix(nearestCube -> getColor(), nearestCube -> getSpecularColor(), isSpecular);
        }
        else
        {
            // Calculate bounced ray

            glm::vec3 point = pixelRay.getOrigin() + (float) rayInfo.t * pixelRay.getDirection();
            glm::vec3 normal = rayInfo.normal;

            glm::vec3 refPoint = point + 0.00001f * normal;

            pixelRay.setOrigin(refPoint);

            // Cosine weighted direction
            glm::vec3 diffuseDirection = glm::normalize(normal + randomVec3Device(seed + (uint32_t) (1645*x + y + 652*i + 8792*frameIndex)));
            glm::vec3 specularDirection = glm::reflect(pixelRay.getDirection(), normal);

            bool isSpecular = nearestSphere->getMetallic() >= randomFloatDevice(seed + (uint32_t) (1645*x + y + 652*i + 8792*frameIndex));
            
            glm::vec3 newDirection = glm::mix(diffuseDirection, specularDirection, (1.0f - nearestSphere->getRoughness()) * isSpecular);
            
            pixelRay.setDirection(newDirection);

            // Calculate lighting

            glm::vec3 emittedLight = nearestSphere -> getEmission();
            light += emittedLight * rayColor;

            rayColor *= glm::mix(nearestSphere -> getColor(), nearestSphere -> getSpecularColor(), isSpecular);
        }
    }

    // R
    accumulatedPixels[index] += light.r;
    // G
    accumulatedPixels[index + 1] += light.g;
    // B
    accumulatedPixels[index + 2] += light.b;
    // A
    accumulatedPixels[index + 3] += 1.0;

    // R
    pixels[index] = glm::clamp(accumulatedPixels[index] / frameIndex, 0.0f, 1.0f);
    // G
    pixels[index + 1] = glm::clamp(accumulatedPixels[index + 1] / frameIndex, 0.0f, 1.0f);
    // B
    pixels[index + 2] = glm::clamp(accumulatedPixels[index + 2] / frameIndex, 0.0f, 1.0f);
    // A
    pixels[index + 3] = glm::clamp(accumulatedPixels[index + 3] / frameIndex, 0.0f, 1.0f);
}

void Render(DisplayInfo& displayInfo, CameraInfo* cameraInfo, std::vector<Sphere>* spheres, std::vector<Cube>* cubes, SkySphere* sky, glm::mat4* viewMatrixInverse, glm::mat4* projectionMatrixInverse, uint32_t& seed)
{
    int width = displayInfo.width;
    int height = displayInfo.height;

    int screenDataSize = sizeof(float) * width * height * 4;

    glm::vec3 cameraPosition = cameraInfo -> cameraPosition;

    int numSpheres = spheres -> size();
    int sphereDataSize = sizeof(Sphere) * numSpheres;

    int numCubes = cubes -> size();
    int cubeDataSize = sizeof(Cube) * numCubes;

    Sphere* sphereData = spheres -> data();
    Cube* cubeData = cubes -> data();

    float* devicePixels = nullptr;
    float* deviceAccumPixels = nullptr;
    Sphere* deviceSpheres = nullptr;
    Cube* deviceCubes = nullptr;
    SkySphere* deviceSky = nullptr;

    sky -> prepareData();

    for (int i = 0; i < numCubes; i++)
    {
        (*cubes)[i].prepareTriangleData();
    }

    cudaMalloc(&devicePixels, screenDataSize);
    cudaMalloc(&deviceAccumPixels, screenDataSize);
    cudaMalloc(&deviceSpheres, sphereDataSize);
    cudaMalloc(&deviceCubes, cubeDataSize);
    cudaMalloc(&deviceSky, sizeof(SkySphere));

    cudaMemcpy(devicePixels, displayInfo.pixels, screenDataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceAccumPixels, displayInfo.accumulatedPixels, screenDataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceSpheres, sphereData, sphereDataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceCubes, cubeData, cubeDataSize, cudaMemcpyHostToDevice);
    gpuErrchk( cudaMemcpy(deviceSky, sky, sizeof(SkySphere), cudaMemcpyHostToDevice) );

    dim3 gridSize = dim3(ceil((float) width / 16), ceil((float) height / 16));
    dim3 blockSize = dim3(16, 16);

    pixelColor <<< gridSize, blockSize >>> (devicePixels, deviceAccumPixels, displayInfo.frameIndex, width, height, deviceSpheres, numSpheres,
                                                                                                                    deviceCubes, numCubes, deviceSky, cameraPosition, *viewMatrixInverse, *projectionMatrixInverse, seed);

    gpuErrchk( cudaMemcpy(displayInfo.pixels, devicePixels, screenDataSize, cudaMemcpyDeviceToHost) );
    cudaMemcpy(displayInfo.accumulatedPixels, deviceAccumPixels, screenDataSize, cudaMemcpyDeviceToHost);

    cudaFree(devicePixels);
    cudaFree(deviceAccumPixels);
    cudaFree(deviceSpheres);
    cudaFree(deviceSky);

    sky -> freeData();

    for (int i = 0; i < numCubes; i++)
    {
        (*cubes)[i].freeTriangleData();
    }

    randomFloat(seed);
}
