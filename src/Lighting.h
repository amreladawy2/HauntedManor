#pragma once
#include <glm/glm.hpp>
#include <vector>

class Shader;

constexpr int MAX_POINT_LIGHTS = 8;

struct DirLight {                      // moonlight
    glm::vec3 direction{-0.3f, -1.0f, -0.2f};
    glm::vec3 color{0.45f, 0.55f, 0.85f};
    float intensity = 0.35f;
};

struct PointLight {                    // fire, candles, lamps
    glm::vec3 position{0.0f};
    glm::vec3 color{1.0f, 0.6f, 0.3f};
    float intensity = 1.0f;
    float range = 6.0f;
    float flicker = 0.0f;              // 0..1 amount of automatic flicker (applied by the game each frame)
};

struct SpotLight {                     // player's flashlight
    glm::vec3 position{0.0f};
    glm::vec3 direction{0.0f, 0.0f, -1.0f};
    glm::vec3 color{1.0f, 0.95f, 0.8f};
    float intensity = 1.6f;
    float innerDeg = 12.0f, outerDeg = 22.0f;
    float range = 14.0f;
    bool on = true;
};

struct LightRig {
    DirLight dir;
    std::vector<PointLight> points;    // at most MAX_POINT_LIGHTS are uploaded
    SpotLight spot;
};

// Uploads the rig to the scene shader (which must be in use). Colors are uploaded pre-multiplied by
// intensity; a SpotLight with on == false is uploaded with zero color. See docs/CONTRACT.md for names.
void uploadLights(const Shader& s, const LightRig& rig);
