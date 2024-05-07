#ifndef OBJECT_H
#define OBJECT_H

#include "../../Common.h"
#include "../Ray.cuh"

typedef struct _Material
{
    glm::vec3 color = glm::vec3(1.0f);
    glm::vec3 specularColor = glm::vec3(1.0f);
    float roughness = 1.0f;
    float metallic = 0.0f;

    glm::vec3 emissionColor = glm::vec3(0.0f);
    float emissionIntensity = 0.0f;

} Material;

typedef struct _RayHitInfo
{
    double t;
    glm::vec3 normal;
    
} RayHitInfo;

class Object
{
protected:
    glm::vec3 position;
    Material material;

public:
    __host__ __device__ glm::vec3 getColor() {return this -> material.color;}
    void setColor(glm::vec3 color) {this -> material.color = color;}

    __host__ __device__ glm::vec3 getSpecularColor() {return this -> material.specularColor;}
    void setSpecularColor(glm::vec3 color) {this -> material.specularColor = color;}

    __host__ __device__ float getRoughness() {return this -> material.roughness;}
    void setRoughness(float roughness) {this -> material.roughness = roughness;}

    __host__ __device__ float getMetallic() {return this -> material.metallic;}
    void setMetallic(float metallic) {this -> material.metallic = metallic;}

    void setEmissionColor(glm::vec3 color) {this -> material.emissionColor = color;}
    void setEmissionIntensity(float intensity) {this -> material.emissionIntensity = intensity;}

    __host__ __device__ glm::vec3 getEmission() {return this -> material.emissionColor * this -> material.emissionIntensity;}

    __host__ __device__ Material getMaterial() {return this -> material;}
    void setMaterial(Material material) {this -> material = material;}
};

#endif // OBJECT_H
