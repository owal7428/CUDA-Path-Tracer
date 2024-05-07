#include "Cube.cuh"

Cube::Cube(float x, float y, float z, float scale_x, float scale_y, float scale_z, float rot_x, float rot_y, float rot_z, float r, float g, float b)
{
    this -> position = glm::vec3(x,y,z);
    
    glm::mat4 modelMatrix = generateModelMatrix(scale_x, scale_y, scale_z, rot_x, rot_y, rot_z);

    Triangle triangle1 = { glm::vec3( modelMatrix * glm::vec4(-0.5f, -0.5f, 0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(0.5f, -0.5f, 0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(0.5f, 0.5f, 0.5f, 1.0f) )};

    Triangle triangle2 = { glm::vec3( modelMatrix * glm::vec4(-0.5f, -0.5f, 0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(0.5f, 0.5f, 0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(-0.5f, 0.5f, 0.5f, 1.0f) )};

    Triangle triangle3 = { glm::vec3( modelMatrix * glm::vec4(0.5f, -0.5f, -0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(-0.5f, -0.5f, -0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(-0.5f, 0.5f, -0.5f, 1.0f) )};

    Triangle triangle4 = { glm::vec3( modelMatrix * glm::vec4(0.5f, -0.5f, -0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(-0.5f, 0.5f, -0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(0.5f, 0.5f, -0.5f, 1.0f) )};

    Triangle triangle5 = { glm::vec3( modelMatrix * glm::vec4(-0.5f, -0.5f, -0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(-0.5f, -0.5f, 0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(-0.5f, 0.5f, 0.5f, 1.0f) )};

    Triangle triangle6 = { glm::vec3( modelMatrix * glm::vec4(-0.5f, -0.5f, -0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(-0.5f, 0.5f, 0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(-0.5f, 0.5f, -0.5f, 1.0f) )};

    Triangle triangle7 = { glm::vec3( modelMatrix * glm::vec4(0.5f, -0.5f, 0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(0.5f, -0.5f, -0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(0.5f, 0.5f, -0.5f, 1.0f) )};

    Triangle triangle8 = { glm::vec3( modelMatrix * glm::vec4(0.5f, -0.5f, 0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(0.5f, 0.5f, -0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(0.5f, 0.5f, 0.5f, 1.0f) )};

    Triangle triangle9 = { glm::vec3( modelMatrix * glm::vec4(-0.5f, 0.5f, 0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(0.5f, 0.5f, 0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(0.5f, 0.5f, -0.5f, 1.0f) )};

    Triangle triangle10 = {glm::vec3( modelMatrix * glm::vec4(-0.5f, 0.5f, 0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(0.5f, 0.5f, -0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(-0.5f, 0.5f, -0.5f, 1.0f) )};

    Triangle triangle11 = {glm::vec3( modelMatrix * glm::vec4(-0.5f, -0.5f, -0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(0.5f, -0.5f, -0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(0.5f, -0.5f, 0.5f, 1.0f) )};

    Triangle triangle12 = {glm::vec3( modelMatrix * glm::vec4(-0.5f, -0.5f, -0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(0.5f, -0.5f, 0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(-0.5f, -0.5f, 0.5f, 1.0f) )};

    triangle1.normal = glm::normalize( glm::cross(triangle1.vertex2 - triangle1.vertex1, triangle1.vertex3 - triangle1.vertex1) );
    triangle2.normal = glm::normalize( glm::cross(triangle2.vertex2 - triangle2.vertex1, triangle2.vertex3 - triangle2.vertex1) );
    triangle3.normal = glm::normalize( glm::cross(triangle3.vertex2 - triangle3.vertex1, triangle3.vertex3 - triangle3.vertex1) );
    triangle4.normal = glm::normalize( glm::cross(triangle4.vertex2 - triangle4.vertex1, triangle4.vertex3 - triangle4.vertex1) );
    triangle5.normal = glm::normalize( glm::cross(triangle5.vertex2 - triangle5.vertex1, triangle5.vertex3 - triangle5.vertex1) );
    triangle6.normal = glm::normalize( glm::cross(triangle6.vertex2 - triangle6.vertex1, triangle6.vertex3 - triangle6.vertex1) );
    triangle7.normal = glm::normalize( glm::cross(triangle7.vertex2 - triangle7.vertex1, triangle7.vertex3 - triangle7.vertex1) );
    triangle8.normal = glm::normalize( glm::cross(triangle8.vertex2 - triangle8.vertex1, triangle8.vertex3 - triangle8.vertex1) );
    triangle9.normal = glm::normalize( glm::cross(triangle9.vertex2 - triangle9.vertex1, triangle9.vertex3 - triangle9.vertex1) );
    triangle10.normal = glm::normalize( glm::cross(triangle10.vertex2 - triangle10.vertex1, triangle10.vertex3 - triangle10.vertex1) );
    triangle11.normal = glm::normalize( glm::cross(triangle11.vertex2 - triangle11.vertex1, triangle11.vertex3 - triangle11.vertex1) );
    triangle12.normal = glm::normalize( glm::cross(triangle12.vertex2 - triangle12.vertex1, triangle12.vertex3 - triangle12.vertex1) );

    triangles.push_back(triangle1);
    triangles.push_back(triangle2);
    triangles.push_back(triangle3);
    triangles.push_back(triangle4);
    triangles.push_back(triangle5);
    triangles.push_back(triangle6);
    triangles.push_back(triangle7);
    triangles.push_back(triangle8);
    triangles.push_back(triangle9);
    triangles.push_back(triangle10);
    triangles.push_back(triangle11);
    triangles.push_back(triangle12);

    this -> material.color = glm::vec3(r,g,b);
}

Cube::Cube(float x, float y, float z, float scale_x, float scale_y, float scale_z, float rot_x, float rot_y, float rot_z, Material material)
{
    this -> position = glm::vec3(x,y,z);
    
    glm::mat4 modelMatrix = generateModelMatrix(scale_x, scale_y, scale_z, rot_x, rot_y, rot_z);

    Triangle triangle1 = { glm::vec3( modelMatrix * glm::vec4(-0.5f, -0.5f, 0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(0.5f, -0.5f, 0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(0.5f, 0.5f, 0.5f, 1.0f) )};

    Triangle triangle2 = { glm::vec3( modelMatrix * glm::vec4(-0.5f, -0.5f, 0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(0.5f, 0.5f, 0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(-0.5f, 0.5f, 0.5f, 1.0f) )};

    Triangle triangle3 = { glm::vec3( modelMatrix * glm::vec4(0.5f, -0.5f, -0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(-0.5f, -0.5f, -0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(-0.5f, 0.5f, -0.5f, 1.0f) )};

    Triangle triangle4 = { glm::vec3( modelMatrix * glm::vec4(0.5f, -0.5f, -0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(-0.5f, 0.5f, -0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(0.5f, 0.5f, -0.5f, 1.0f) )};

    Triangle triangle5 = { glm::vec3( modelMatrix * glm::vec4(-0.5f, -0.5f, -0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(-0.5f, -0.5f, 0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(-0.5f, 0.5f, 0.5f, 1.0f) )};

    Triangle triangle6 = { glm::vec3( modelMatrix * glm::vec4(-0.5f, -0.5f, -0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(-0.5f, 0.5f, 0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(-0.5f, 0.5f, -0.5f, 1.0f) )};

    Triangle triangle7 = { glm::vec3( modelMatrix * glm::vec4(0.5f, -0.5f, 0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(0.5f, -0.5f, -0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(0.5f, 0.5f, -0.5f, 1.0f) )};

    Triangle triangle8 = { glm::vec3( modelMatrix * glm::vec4(0.5f, -0.5f, 0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(0.5f, 0.5f, -0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(0.5f, 0.5f, 0.5f, 1.0f) )};

    Triangle triangle9 = { glm::vec3( modelMatrix * glm::vec4(-0.5f, 0.5f, 0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(0.5f, 0.5f, 0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(0.5f, 0.5f, -0.5f, 1.0f) )};

    Triangle triangle10 = {glm::vec3( modelMatrix * glm::vec4(-0.5f, 0.5f, 0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(0.5f, 0.5f, -0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(-0.5f, 0.5f, -0.5f, 1.0f) )};

    Triangle triangle11 = {glm::vec3( modelMatrix * glm::vec4(-0.5f, -0.5f, -0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(0.5f, -0.5f, -0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(0.5f, -0.5f, 0.5f, 1.0f) )};

    Triangle triangle12 = {glm::vec3( modelMatrix * glm::vec4(-0.5f, -0.5f, -0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(0.5f, -0.5f, 0.5f, 1.0f) ),
                            glm::vec3( modelMatrix * glm::vec4(-0.5f, -0.5f, 0.5f, 1.0f) )};
    
    triangle1.normal = glm::normalize( glm::cross(triangle1.vertex2 - triangle1.vertex1, triangle1.vertex3 - triangle1.vertex1) );
    triangle2.normal = glm::normalize( glm::cross(triangle2.vertex2 - triangle2.vertex1, triangle2.vertex3 - triangle2.vertex1) );
    triangle3.normal = glm::normalize( glm::cross(triangle3.vertex2 - triangle3.vertex1, triangle3.vertex3 - triangle3.vertex1) );
    triangle4.normal = glm::normalize( glm::cross(triangle4.vertex2 - triangle4.vertex1, triangle4.vertex3 - triangle4.vertex1) );
    triangle5.normal = glm::normalize( glm::cross(triangle5.vertex2 - triangle5.vertex1, triangle5.vertex3 - triangle5.vertex1) );
    triangle6.normal = glm::normalize( glm::cross(triangle6.vertex2 - triangle6.vertex1, triangle6.vertex3 - triangle6.vertex1) );
    triangle7.normal = glm::normalize( glm::cross(triangle7.vertex2 - triangle7.vertex1, triangle7.vertex3 - triangle7.vertex1) );
    triangle8.normal = glm::normalize( glm::cross(triangle8.vertex2 - triangle8.vertex1, triangle8.vertex3 - triangle8.vertex1) );
    triangle9.normal = glm::normalize( glm::cross(triangle9.vertex2 - triangle9.vertex1, triangle9.vertex3 - triangle9.vertex1) );
    triangle10.normal = glm::normalize( glm::cross(triangle10.vertex2 - triangle10.vertex1, triangle10.vertex3 - triangle10.vertex1) );
    triangle11.normal = glm::normalize( glm::cross(triangle11.vertex2 - triangle11.vertex1, triangle11.vertex3 - triangle11.vertex1) );
    triangle12.normal = glm::normalize( glm::cross(triangle12.vertex2 - triangle12.vertex1, triangle12.vertex3 - triangle12.vertex1) );
    
    triangles.push_back(triangle1);
    triangles.push_back(triangle2);
    triangles.push_back(triangle3);
    triangles.push_back(triangle4);
    triangles.push_back(triangle5);
    triangles.push_back(triangle6);
    triangles.push_back(triangle7);
    triangles.push_back(triangle8);
    triangles.push_back(triangle9);
    triangles.push_back(triangle10);
    triangles.push_back(triangle11);
    triangles.push_back(triangle12);

    this -> material = material;
}

glm::mat4 Cube::generateModelMatrix(float scale_x, float scale_y, float scale_z, float rot_x, float rot_y, float rot_z)
{
    glm::vec3 scale = glm::vec3(scale_x, scale_y, scale_z);

    // Build quaternions for x, y, and z axis rotations, then combine
    glm::quat quat_x = glm::angleAxis(glm::radians(rot_x), glm::vec3(1,0,0));
    glm::quat quat_y = glm::angleAxis(glm::radians(rot_y), glm::vec3(0,1,0));
    glm::quat quat_z = glm::angleAxis(glm::radians(rot_z), glm::vec3(0,0,1));

    glm::quat rotation = quat_x * quat_y * quat_z;

    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, position);
    model = model * glm::toMat4(rotation);
    model = glm::scale(model, scale);

    return model;
}

__device__ RayHitInfo Cube::checkHit(Ray* ray)
{
    RayHitInfo rayInfo = {1e100, glm::vec3(0.0f)};

    for (int i = 0; i < 12; i++)
    {
        RayHitInfo temp = checkTriangleHit(ray, &deviceTriangles[i]);

        if (temp.t > 0.0f && temp.t < rayInfo.t)
        {
            rayInfo = temp;
        }
    }

    return rayInfo;
}

__device__ RayHitInfo Cube::checkTriangleHit(Ray* ray, Triangle* triangle)
{
    glm::vec3 edgeAB = triangle->vertex2 - triangle->vertex1;
    glm::vec3 edgeAC = triangle->vertex3 - triangle->vertex1;

    glm::vec3 normalVec = glm::cross(edgeAB, edgeAC);

    glm::vec3 AO = ray->getOrigin() - triangle->vertex1;
    glm::vec3 DAO = glm::cross(AO, ray->getDirection());

    double determinant = -1.0f * (double) glm::dot(ray->getDirection(), normalVec);

    if (determinant <= 1e-6)
        return {-1.0f, glm::vec3(0.0f)};

    double invDeterminant = 1.0f / determinant;

    double t = glm::dot(AO, normalVec) * invDeterminant;

    if (t <= 0)
        return {-1.0f, glm::vec3(0.0f)};

    double u = glm::dot(edgeAC, DAO) * invDeterminant;
    double v = -1.0f * glm::dot(edgeAB, DAO) * invDeterminant;
    double w = 1 - u - v;

    if (u <= 0 || v <= 0 || w <= 0)
        return {-1.0f, glm::vec3(0.0f)};

    return {t, triangle->normal};
}

void Cube::prepareTriangleData()
{
    unsigned int dataSize = sizeof(Triangle) * triangles.size();

    Triangle* triangleData = triangles.data();
    
    cudaMalloc(&deviceTriangles, dataSize);
    cudaMemcpy(deviceTriangles, triangleData, dataSize, cudaMemcpyHostToDevice);
}

void Cube::freeTriangleData()
{
    cudaFree(deviceTriangles);
}
