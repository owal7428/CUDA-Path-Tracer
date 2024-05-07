#ifndef SPHERE_H
#define SPHERE_H

#include "Object.cuh"

class Sphere : public Object
{
private:    
    float radius;

public:
    Sphere(float x, float y, float z, float radius, float r=1, float g=1, float b=1)
    {
        this -> position = glm::vec3(x,y,z);
        this -> radius = radius;

        this -> material.color = glm::vec3(r,g,b);
    }

    Sphere(float x, float y, float z, float radius, Material material)
    {
        this -> position = glm::vec3(x,y,z);
        this -> radius = radius;

        this -> material = material;
    }

    // Return -1 if no hit, else return closest t value
    __device__ RayHitInfo checkHit(Ray* ray);
    __device__ glm::vec3 getNormal(glm::vec3& point);

    __host__ __device__ float getRadius() {return radius;}
};

#endif // SPHERE_H
