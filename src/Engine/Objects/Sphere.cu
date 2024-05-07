#include "Sphere.cuh"

__device__ RayHitInfo Sphere::checkHit(Ray* ray)
{
    glm::vec3 distance = ray -> getOrigin() - position;
    glm::vec3 dir = ray -> getDirection();

    double A = glm::dot(dir, dir);

    double B = 2.0f * glm::dot(dir, distance);

    double C = glm::dot(distance, distance) - (radius * radius);

    double discr = (B * B) - (4 * A * C);

    if (discr < 0)
        return {-1.0f, glm::vec3(0.0f)};
    
    double t1 = ((-1 * B) + glm::sqrt(discr)) / (2*A);
    double t2 = ((-1 * B) - glm::sqrt(discr)) / (2*A);

    // Return the closest of the two
    double t = (t1 < t2) ? t1 : t2;

    glm::vec3 point = ray -> getOrigin() + (float) t * dir;
    glm::vec3 normal = glm::normalize(point - position);

    return {t, normal};
}

__device__ glm::vec3 Sphere::getNormal(glm::vec3& point)
{
    return glm::normalize(point - position);
}
