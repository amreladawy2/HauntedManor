#pragma once
#include <GL/glew.h>
#include <string>

// Loads an image (png/jpg/tga/bmp via stb_image) into a mipmapped, repeat-wrapped 2D texture.
// Results are cached by absolute path. On failure prints a warning and returns a magenta/black checker.
GLuint loadTexture(const std::string& path);

// 1x1 white texture, used for untextured materials.
GLuint whiteTexture();
