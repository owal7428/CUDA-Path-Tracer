#ifndef GL_INCLUDES_H
#define GL_INCLUDES_H

//#define USEGLEW

#ifdef USEGLEW
#include <GL/glew.h>
#endif // USEGLEW

#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>

#ifdef __APPLE__

#include <OpenGL/glu.h>
#include <OpenGL/gl.h>
// Tell Xcode IDE to not gripe about OpenGL deprecation
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#else

#if defined(_WIN32) || defined(WIN32)

#include <windows.h>

#endif

#include <GL/glu.h>
#include <GL/gl.h>

#endif // __APPLE__

//  Default resolution
//  For Retina displays compile with -DRES=2
#ifndef RES
#define RES 1
#endif // RES

#endif // GL_INCLUDES_H
