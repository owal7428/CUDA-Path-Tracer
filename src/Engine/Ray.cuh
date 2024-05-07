#ifndef RAY_H
#define RAY_H

#include "../Common.h"

class Ray
{
private:
    glm::vec3 origin;
    glm::vec3 direction;

public:
    // Generate ray from normalized device coordinates
     __device__ Ray(glm::vec3 origin, glm::vec2 coords, glm::mat4* viewMatrixInverse, glm::mat4* projectionMatrixInverse)
    {
        glm::vec4 target = *projectionMatrixInverse * glm::vec4(coords, 1.0f, 1.0f);
        glm::vec4 newDirection = *viewMatrixInverse * glm::vec4(glm::normalize(glm::vec3(target) / target.w), 0.0f);

        this -> origin = origin;
        this -> direction = glm::vec3(newDirection);
    }

    // Generate ray with defined direction in world space
    __device__ Ray(glm::vec3 origin, glm::vec3 direction)
    {
        this -> origin = origin;
        this -> direction = direction;
    }

    __device__ glm::vec3 getOrigin() {return origin;}
    __device__ glm::vec3 getDirection() {return direction;}

    __device__ void setOrigin(glm::vec3 origin) {this -> origin = origin;}
    __device__ void setDirection(glm::vec3 direction) {this -> direction = direction;}
};

#endif // RAY_H
