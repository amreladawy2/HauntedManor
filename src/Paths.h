#pragma once
#include <string>

// Resolves a project-relative path such as "assets/models/lib_desk.obj" or "shaders/scene.vert".
// Search order: current dir, up to 3 parent dirs, then PROJECT_ROOT (compile definition set by CMake).
// Returns the input unchanged when nothing matches (the caller then reports the load failure).
std::string resolvePath(const std::string& relative);
