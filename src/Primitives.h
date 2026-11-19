#pragma once
#include "Model.h"

// Quad in the XY plane centred at the origin, normal +Z, size w x h.
// UVs span [0,uRepeat] x [0,vRepeat] so tiled textures repeat.
Mesh makeQuad(float w, float h, float uRepeat, float vRepeat, GLuint texture);

// Axis-aligned box centred at the origin with outward normals; UVs tile at uvPerMeter repeats per metre.
Mesh makeBox(const glm::vec3& size, float uvPerMeter, GLuint texture);
