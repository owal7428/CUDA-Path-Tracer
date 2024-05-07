#ifndef DIRECTIONAL_LIGHT_H
#define DIRECTIONAL_LIGHT_H

#include "../Common.h"

class DirectionalLight
{
private:
    glm::vec3 direction;
    glm::vec3 color;

public:
    /* Constructor for directional light.
    *  @param xyz direction of light
       @param rgb color of light */
    DirectionalLight(float x, float y, float z, float r, float g, float b)
    {
        this -> direction = glm::normalize(glm::vec3(x,y,z));
        this -> color = glm::vec3(r,g,b);
    }

    void changeColor(float r, float g, float b) {color = glm::vec3(r, g, b);}
    void setDirection(float x, float y, float z) {direction = glm::vec3(x,y,z);}

    inline glm::vec3 getDirection() {return direction;}
    inline glm::vec3 getColor() {return color;}
};

#endif // DIRECTIONAL_LIGHT_H
