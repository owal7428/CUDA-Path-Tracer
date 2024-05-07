#ifndef CUBE_H
#define CUBE_H

#include "Object.cuh"

typedef struct _Triangle
{
    glm::vec3 vertex1;
    glm::vec3 vertex2;
    glm::vec3 vertex3;

    glm::vec3 normal;

} Triangle;

class Cube : public Object
{
private:
    std::vector<Triangle> triangles;
    Triangle* deviceTriangles;

    glm::mat4 generateModelMatrix(float scale_x, float scale_y, float scale_z, float rot_x, float rot_y, float rot_z);
    __device__ RayHitInfo checkTriangleHit(Ray* ray, Triangle* triangle);

public:
    Cube(float x, float y, float z, float scale_x, float scale_y, float scale_z, float rot_x, float rot_y, float rot_z, float r=1, float g=1, float b=1);
    Cube(float x, float y, float z, float scale_x, float scale_y, float scale_z, float rot_x, float rot_y, float rot_z, Material material);

    // Return -1 if no hit, else return closest t value
    __device__ RayHitInfo checkHit(Ray* ray);

    void prepareTriangleData();
    void freeTriangleData();
};

#endif // CUBE_H
