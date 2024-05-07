#ifndef SKY_SPHERE_H
#define SKY_SPHERE_H

#include "../../Common.h"
#include "../Ray.cuh"

class SkySphere
{
private:
    float* data;
    float* deviceData;
    int image_width, image_height, image_channels;

    bool isEnabled = true;
    bool isDark = false;

public:
    SkySphere(const char* filepath);
    ~SkySphere();

    void prepareData();
    void freeData();

    void changeTexture(const char* filepath);

    inline void Enable() {isEnabled = true;}
    inline void Disable() {isEnabled = false;}
    inline void ToggleDarken() {isDark = !isDark;}

    __device__ glm::vec3 getSkyColor(Ray* ray);
    __device__ inline bool getIsEnabled() {return isEnabled;}

};

#endif // SKY_SPHERE_H
