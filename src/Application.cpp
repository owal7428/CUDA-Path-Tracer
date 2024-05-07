#include "Common.h"
#include "GLIncludes.h"
#include "Utility/util.hpp"

#include "Engine/Ray.cuh"
#include "Engine/Renderer.cuh"
#include "Engine/Objects/Cube.cuh"
#include "Engine/Objects/Sphere.cuh"
#include "Engine/Objects/SkySphere.cuh"

// Current states of the arrow keys

int leftKeyDown = 0;
int rightKeyDown = 0;
int upKeyDown = 0;
int downKeyDown = 0;

// Current states of the wasd keys

int wKeyDown = 0;
int aKeyDown = 0;
int sKeyDown = 0;
int dKeyDown = 0;

int bKeyDown = 0;
int cKeyDown = 0;

// Mouse movement

int mouseMoveX = 0;
int mouseMoveY = 0;

// Display parameters

int width = 0;
int height = 0;

float asp = 1;
float fov = 45;
float zNear = 1;
float zFar = 30;

// Camera parameters

glm::vec3 cameraPosition = glm::vec3(0.0f, 0.25f, 0.0f);
glm::vec3 cameraDirection = glm::vec3(-1.0f, 0.0f, 0.0f);

float theta = -90;
float phi = 0;

glm::mat4 viewMatrixInverse;
glm::mat4 projectionMatrixInverse;

// Display buffers

float* pixels = nullptr;
float* accumulatedPixels = nullptr;

uint32_t seed = 0x58f0a5c;

int frameIndex = 1;

// Texture path macros

#define COBBLESTONE_HDR "resources/cobblestone_street_night_2k.hdr"
#define INDUSTRIAL_HDR  "resources/industrial_pipe_and_valve_01_2k.hdr"
#define GOLDEN_BAY_HDR  "resources/golden_bay_2k.hdr"
#define SUNSET_HDR      "resources/sunset_jhbcentral_2k.hdr"
#define GARDEN_HDR      "resources/symmetrical_garden_02_2k.hdr"

char const* skyTextures[5] = {COBBLESTONE_HDR, INDUSTRIAL_HDR, GOLDEN_BAY_HDR, SUNSET_HDR, GARDEN_HDR};
int currentTexture = 0;

int currentScene = 0;

void calculateView()
{
    viewMatrixInverse = glm::inverse(glm::lookAt(cameraPosition, cameraPosition + cameraDirection, glm::vec3(0.0f, 1.0f, 0.0f)));
}

void calculateProjection()
{
    projectionMatrixInverse = glm::inverse(glm::perspective(glm::radians(fov), asp, zNear, zFar));
}

void draw(SDL_Window* window, std::vector<Sphere>* spheres, std::vector<Cube>* cubes, SkySphere* sky)
{
    // Clear the screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (frameIndex == 1)
        memset(accumulatedPixels, 0, width * height * 4 * sizeof(float));

    DisplayInfo displayInfo = 
    {
        width,
        height,
        pixels,
        accumulatedPixels,
        frameIndex
    };

    CameraInfo cameraInfo =
    {
        cameraPosition,
        cameraDirection
    };

    Render(displayInfo, &cameraInfo, spheres, cubes, sky, &viewMatrixInverse, &projectionMatrixInverse, seed);
    
    pixels = displayInfo.pixels;
    accumulatedPixels = displayInfo.accumulatedPixels;
    frameIndex = displayInfo.frameIndex;

    glDrawPixels(width, height, GL_RGBA, GL_FLOAT, pixels);

    frameIndex++;

    glFlush();
    SDL_GL_SwapWindow(window);
}

void keyDown(SDL_Scancode code)
{
    switch (code)
    {      
        case SDL_SCANCODE_W:
            wKeyDown = 1;
            break;
        
        case SDL_SCANCODE_A:
            aKeyDown = 1;
            break;
        
        case SDL_SCANCODE_S:
            sKeyDown = 1;
            break;
        
        case SDL_SCANCODE_D:
            dKeyDown = 1;
            break;
        
        case SDL_SCANCODE_RIGHT:
            rightKeyDown = 1;
            break;
        
        case SDL_SCANCODE_LEFT:
            leftKeyDown = 1;
            break;
        
        case SDL_SCANCODE_UP:
            upKeyDown = 1;
            break;
        
        case SDL_SCANCODE_DOWN:
            downKeyDown = 1;
            break;
        
        case SDL_SCANCODE_B:
            bKeyDown = 1;
            break;
        
        case SDL_SCANCODE_C:
            cKeyDown = 1;
            break;
        
        default:
            break;
    }
}

void keyUp(SDL_Scancode code)
{
    switch (code)
    {
        case SDL_SCANCODE_W:
            wKeyDown = 0;
            break;
        
        case SDL_SCANCODE_A:
            aKeyDown = 0;
            break;
        
        case SDL_SCANCODE_S:
            sKeyDown = 0;
            break;
        
        case SDL_SCANCODE_D:
            dKeyDown = 0;
            break;
        
        case SDL_SCANCODE_RIGHT:
            rightKeyDown = 0;
            break;
        
        case SDL_SCANCODE_LEFT:
            leftKeyDown = 0;
            break;
        
        case SDL_SCANCODE_UP:
            upKeyDown = 0;
            break;
        
        case SDL_SCANCODE_DOWN:
            downKeyDown = 0;
            break;
        
        default:
            break;
    }
}

void checkMouseInput(int xRel, int yRel)
{
    mouseMoveX += xRel;
    mouseMoveY += yRel;
}

/* This function is used for smooth rotation;
*  Checks for key presses every 10 milliseconds and updates rotation accordingly. */
void Update(double deltaTime)
{
    bool hasChanged = false;

    float newTheta = theta;
    float newPhi = phi;

    if (rightKeyDown || leftKeyDown)
    {
        if (rightKeyDown && !leftKeyDown)
        {
            newTheta -= 1;
            hasChanged = true;
        }
        
        else if (leftKeyDown && !rightKeyDown)
        {
            newTheta += 1;
            hasChanged = true;
        }
    }

    if (upKeyDown || downKeyDown)
    {
        if (upKeyDown && !downKeyDown)
        {
            newPhi += 1;
            hasChanged = true;
        }
        
        else if (downKeyDown && !upKeyDown)
        {
            newPhi -= 1;
            hasChanged = true;
        }
    }

    if (mouseMoveX != 0 || mouseMoveY != 0)
    {
        newTheta -= mouseMoveX * 2 * deltaTime;
        newPhi -= mouseMoveY * 2 * deltaTime;

        mouseMoveX = 0;
        mouseMoveY = 0;

        hasChanged = true;
    }

    // Only calculate this if the values are different

    if (newTheta != theta || newPhi != phi)
    {
        theta = newTheta;
        phi = newPhi;

        while (theta >= 360)
            theta -= 360;
        
        while (theta < 0)
            theta += 360;
        
        if (phi > 89)
            phi = 89;
        
        else if (phi < -89)
            phi = -89;
        
        // REMEMBER: +x points left, +y points up, +z points forward

        float newX = glm::cos(glm::radians(phi)) * glm::cos(glm::radians(theta - 90));
        float newY = glm::sin(glm::radians(phi));
        float newZ = glm::cos(glm::radians(phi)) * glm::sin(glm::radians(theta + 90));

        cameraDirection = glm::vec3(newX, newY, newZ);
    }

    if (wKeyDown || sKeyDown)
    {
        if (wKeyDown && !sKeyDown)
        {
            cameraPosition += cameraDirection * (float) deltaTime;
            hasChanged = true;
        }
        
        else if (sKeyDown && !wKeyDown)
        {
            cameraPosition -= cameraDirection * (float) deltaTime;
            hasChanged = true;
        }
    }

    // Line on unit circle 90 degrees behind will always be perpendicular

    float xOffset = glm::cos(glm::radians(theta - 180));
    float zOffset = glm::sin(glm::radians(theta));

    if (dKeyDown || aKeyDown)
    {
        if (dKeyDown && !aKeyDown)
        {
            cameraPosition += glm::vec3(xOffset, 0.0f, zOffset) * (float) deltaTime;
            hasChanged = true;
        }

        else if (aKeyDown && !dKeyDown)
        {
            cameraPosition -= glm::vec3(xOffset, 0.0f, zOffset) * (float) deltaTime;
            hasChanged = true;
        }
    }

    if (hasChanged)
    {
        calculateView();
        frameIndex = 1;
    }
}

void reshape(SDL_Window* window)
{
    SDL_GetWindowSize(window, &width, &height);

    asp = (height > 0) ? (double) width/height : 1;

    glViewport(0, 0, RES * width, RES * height);

    delete pixels;
    pixels = new float [4 * width * height];
    
    delete accumulatedPixels;
    accumulatedPixels = new float [4 * width * height];

    frameIndex = 1;

    calculateProjection();
}

int main(int argc, char* argv[]) 
{
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window* window = SDL_CreateWindow("OwenAllison_FinalProject", 
                                          SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 1920, 1080, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);

    SDL_ShowCursor(SDL_DISABLE);
    SDL_SetRelativeMouseMode(SDL_TRUE);

    bool catchMouse = true;

    SDL_GL_CreateContext(window);                      
    
    #ifdef USEGLEW
    //  Initialize GLEW
    if (glewInit()!=GLEW_OK) fprintf(stderr, "Error initializing GLEW\n");
    #endif
    
    reshape(window);

    calculateView();

    // Create Objects
    std::vector<Sphere> spheres_scene1;
    std::vector<Cube> cubes_scene1;

    SkySphere skySphere = SkySphere(COBBLESTONE_HDR);

    // Scene 1
    Sphere sun = Sphere(-6, 1, 0, 3, 0.8, 0.2, 0.05);
    Sphere bigSphere = Sphere(0, -10, 0, 10, 0.85, 0.85, 0.85);
    Sphere littleSphere = Sphere(0, 0.25, 0.5, 0.25, 1.0, 1.0, 1.0);
    Sphere littleSphere2 = Sphere(0, 0.25, -0.5, 0.25, 0.1, 1.0, 0.2);
    Sphere shinySphere = Sphere(-1, 0.225, 0, 0.25, 0.98, 0.98, 0.98);

    Cube testCube = Cube(0, 0.3, -2, 1, 1, 1, 0, 0, 0, 1.0f, 0.2f, 0.2f);

    testCube.setRoughness(0.1f);
    testCube.setMetallic(1.0f);
    testCube.setSpecularColor(glm::vec3(1.0f, 0.6f, 0.6f));

    sun.setEmissionColor(sun.getColor());
    sun.setEmissionIntensity(15.0f);

    shinySphere.setRoughness(0.0f);
    shinySphere.setMetallic(1.0f);

    //spheres.push_back(sun);
    spheres_scene1.push_back(bigSphere);
    spheres_scene1.push_back(littleSphere);
    spheres_scene1.push_back(littleSphere2);
    spheres_scene1.push_back(shinySphere);

    cubes_scene1.push_back(testCube);

    std::vector<Sphere> spheres_scene2;
    std::vector<Cube> cubes_scene2;

    Sphere littleSphere1_scene2 = Sphere(0.0, 0.3, 0.0, 0.25);
    Sphere littleSphere2_scene2 = Sphere(0.0, 0.3, 0.6, 0.25);
    Sphere littleSphere3_scene2 = Sphere(0.0, 0.3, 1.2, 0.25);
    Sphere littleSphere4_scene2 = Sphere(0.0, 0.3, -0.6, 0.25);
    Sphere littleSphere5_scene2 = Sphere(0.0, 0.3, -1.2, 0.25);
    Sphere emissiveSphere1_scene2 = Sphere(-3.0, 1.75, 1.2, 1.5, 0.0, 0.0, 0.0);
    Sphere emissiveSphere2_scene2 = Sphere(1.0, 1.5, 2.5, 0.75, 0.0, 0.0, 0.0);

    Cube floor_scene2 = Cube(0.0, 0.0, 0.0, 2.5, 0.1, 5.0, 0, 0, 0);

    littleSphere1_scene2.setRoughness(0.0f);
    littleSphere1_scene2.setMetallic(1.0f);

    littleSphere2_scene2.setRoughness(0.15f);
    littleSphere2_scene2.setMetallic(0.05f);

    littleSphere3_scene2.setRoughness(0.3f);
    littleSphere3_scene2.setMetallic(0.9f);

    littleSphere5_scene2.setRoughness(0.05f);
    littleSphere5_scene2.setMetallic(1.0f);
    littleSphere5_scene2.setSpecularColor(glm::vec3(0.4f, 1.0f, 0.4f));

    emissiveSphere1_scene2.setEmissionColor(glm::vec3(0.05f, 0.05f, 1.0f));
    emissiveSphere1_scene2.setEmissionIntensity(3.0f);

    emissiveSphere2_scene2.setEmissionColor(glm::vec3(0.8, 0.8, 0.05));
    emissiveSphere2_scene2.setEmissionIntensity(3.0f);

    spheres_scene2.push_back(littleSphere1_scene2);
    spheres_scene2.push_back(littleSphere2_scene2);
    spheres_scene2.push_back(littleSphere3_scene2);
    spheres_scene2.push_back(littleSphere4_scene2);
    spheres_scene2.push_back(littleSphere5_scene2);
    spheres_scene2.push_back(emissiveSphere1_scene2);
    spheres_scene2.push_back(emissiveSphere2_scene2);

    cubes_scene2.push_back(floor_scene2);

    std::vector<Sphere> spheres_scene3;
    std::vector<Cube> cubes_scene3;

    // Scene 3
    Sphere littleSphere_scene3 = Sphere(0.25, 0.375, 0, 0.25, 1.0, 1.0, 1.0);
    Sphere littleSphere2_scene3 = Sphere(-0.1, 0.3, 0, 0.10, 1.0, 1.0, 1.0);

    Cube floor_scene3 = Cube(0.0, 0.0, 0.0, 1.5, 0.25, 1.5, 0, 0, 0);
    Cube roof_scene3 = Cube(0.0, 1.5, 0.0, 1.5, 0.25, 1.5, 0, 0, 0);
    Cube wall1_scene3 = Cube(0.75, 0.75, 0.0, 0.05, 1.25, 1.5, 0, 0, 0);
    Cube wall2_scene3 = Cube(-0.75, 0.75, 0.0, 0.05, 1.25, 1.5, 0, 0, 0);
    Cube wall3_scene3 = Cube(0.0, 0.75, 0.75, 0.05, 1.25, 1.5, 0, 90, 0);
    Cube wall4_scene3 = Cube(0.0, 0.75, -0.75, 0.05, 1.25, 1.5, 0, 90, 0);
    Cube light_scene3 = Cube(0.0, 1.4, 0.0, 0.25, 0.25, 0.25, 0, 0, 0);

    light_scene3.setEmissionColor(glm::vec3(1.0f));
    light_scene3.setEmissionIntensity(5.0f);

    littleSphere2_scene3.setEmissionColor(glm::vec3(1.0f, 0.2f, 0.9f));
    littleSphere2_scene3.setEmissionIntensity(1.0f);

    wall1_scene3.setRoughness(0.005f);
    wall1_scene3.setMetallic(1.0f);
    wall1_scene3.setSpecularColor(glm::vec3(1.0f, 0.6f, 0.6f));

    wall2_scene3.setRoughness(0.005f);
    wall2_scene3.setMetallic(1.0f);
    wall2_scene3.setSpecularColor(glm::vec3(0.6f, 0.6f, 1.0f));

    wall3_scene3.setRoughness(0.005f);
    wall3_scene3.setMetallic(1.0f);

    wall4_scene3.setRoughness(0.005f);
    wall4_scene3.setMetallic(1.0f);

    spheres_scene3.push_back(littleSphere_scene3);
    spheres_scene3.push_back(littleSphere2_scene3);

    cubes_scene3.push_back(floor_scene3);
    cubes_scene3.push_back(roof_scene3);
    cubes_scene3.push_back(wall1_scene3);
    cubes_scene3.push_back(wall2_scene3);
    cubes_scene3.push_back(wall3_scene3);
    cubes_scene3.push_back(wall4_scene3);
    cubes_scene3.push_back(light_scene3);

    std::vector<Sphere> spheres_scene4;
    std::vector<Cube> cubes_scene4;

    // Scene 4
    Sphere bigSphere_scene4 = Sphere(0, -10, 0, 10, 0.85, 0.85, 0.85);
    Sphere littleSphere_scene4 = Sphere(0, 0.25, 0.5, 0.25, 1.0, 1.0, 1.0);
    Sphere littleSphere2_scene4 = Sphere(0, 0.25, -0.5, 0.25, 0.1, 1.0, 0.2);
    Sphere shinySphere_scene4 = Sphere(-1, 0.225, 0, 0.25, 0.98, 0.98, 0.98);

    Cube testCube_scene4 = Cube(0, 0.3, -2, 1, 1, 1, 0, 0, 0, 1.0f, 0.2f, 0.2f);

    testCube_scene4.setRoughness(0.0f);
    testCube_scene4.setMetallic(1.0f);
    testCube_scene4.setSpecularColor(glm::vec3(1.0f, 0.6f, 0.6f));

    littleSphere_scene4.setRoughness(0.3f);
    littleSphere_scene4.setMetallic(1.0f);

    littleSphere2_scene4.setRoughness(0.0f);
    littleSphere2_scene4.setMetallic(0.25f);

    shinySphere_scene4.setRoughness(0.0f);
    shinySphere_scene4.setMetallic(1.0f);

    spheres_scene4.push_back(bigSphere_scene4);
    spheres_scene4.push_back(littleSphere_scene4);
    spheres_scene4.push_back(littleSphere2_scene4);
    spheres_scene4.push_back(shinySphere_scene4);

    cubes_scene4.push_back(testCube_scene4);

    int run = 1;
    double time = 0;

    // Event loop
    while (run)
    {
        double newTime = SDL_GetTicks64()/1000.0;
        double deltaTime = newTime - time;

        // Do this every 0.01 seconds (10 ms)
        if (deltaTime >= 0.01)
        {
            time = newTime;
            Update(deltaTime);
        }

        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            switch (event.type)
            {
                case SDL_WINDOWEVENT:
                    if (event.window.event == SDL_WINDOWEVENT_RESIZED)
                    {
                        int width = event.window.data1;
                        int height = event.window.data2;
                        SDL_SetWindowSize(window, width, height);
                        reshape(window);
                    }
                    break;
                case SDL_QUIT:
                    run = 0;
                    break;
                case SDL_KEYDOWN:
                    // Exit event loop if escape key is pressed
                    if (event.key.keysym.scancode == SDL_SCANCODE_ESCAPE)
                        run = 0;
                    else if (event.key.keysym.scancode == SDL_SCANCODE_F1)
                    {
                        SDL_ShowCursor(SDL_ENABLE);
                        SDL_SetRelativeMouseMode(SDL_FALSE);
                        catchMouse = false;
                    }
                    else if (event.key.keysym.scancode == SDL_SCANCODE_F2)
                    {
                        SDL_ShowCursor(SDL_DISABLE);
                        SDL_SetRelativeMouseMode(SDL_TRUE);
                        catchMouse = true;
                    }
                    else
                        keyDown(event.key.keysym.scancode);
                    break;
                case SDL_KEYUP:
                    keyUp(event.key.keysym.scancode);
                    break;
                case SDL_MOUSEMOTION:
                    if (catchMouse)
                        checkMouseInput(event.motion.xrel, event.motion.yrel);
                default:
                    break;
            }
        }

        if (bKeyDown)
        {
            currentTexture++;
            currentTexture %= 5;

            skySphere.changeTexture(skyTextures[currentTexture]);

            frameIndex = 1;

            bKeyDown = 0;
        }

        if (cKeyDown)
        {
            currentScene++;
            currentScene %= 4;

            frameIndex = 1;

            cKeyDown = 0;
        }

        if (currentScene == 0)
        {
            skySphere.Disable();
            draw(window, &spheres_scene1, &cubes_scene1, &skySphere);
        }

        if (currentScene == 1)
        {
            skySphere.Enable();
            skySphere.ToggleDarken();
            draw(window, &spheres_scene2, &cubes_scene2, &skySphere);
            skySphere.ToggleDarken();
        }

        if (currentScene == 2)
        {
            skySphere.Disable();
            draw(window, &spheres_scene3, &cubes_scene3, &skySphere);
        }

        if (currentScene == 3)
        {
            skySphere.Enable();
            draw(window, &spheres_scene4, &cubes_scene4, &skySphere);
        }
    }

    SDL_Quit();
    delete pixels;
    delete accumulatedPixels;
    return 0;
}
