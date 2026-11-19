#pragma once
#include <glm/glm.hpp>
#include <string>
#include <vector>
#include "Lighting.h"

class Model;
class ModelCache;

// Interaction animation kinds; combine with bitwise OR.
enum AnimFlags : unsigned {
    ANIM_ROTATE    = 1u << 0,   // continuous spin about rotAxis at rotSpeedDeg deg/s
    ANIM_SWING     = 1u << 1,   // sinusoidal rotation about rotAxis: swingAmpDeg * sin(2*pi*swingFreq*t)
    ANIM_TRANSLATE = 1u << 2,   // eases from 0 to translateBy in the first 40% of duration, then holds
    ANIM_SCALE     = 1u << 3,   // pulses between 1 and scaleTo: 1 + (scaleTo-1)*(0.5-0.5*cos(2*pi*1.0*t))
    ANIM_COLOR     = 1u << 4,   // tint/emissive ease to colorTo/emissiveTo in the first 30% of duration
};

// What an object does on its own while horror mode is on (no interaction needed).
enum class HorrorAnim { None, Sway, Float, Rattle, Rock, Spin, StareAtPlayer, Pulse };

struct Interaction {
    bool enabled = false;
    std::string prompt;                 // HUD hint, e.g. "Spin the globe"
    float range = 2.0f;                 // max XZ distance player->object position (metres)
    float viewAngleDeg = 40.0f;         // max angle between player's horizontal facing and direction to object
    glm::vec3 camPos{0.0f};             // world-space camera position while interacting
    glm::vec3 camTarget{0.0f};          // world-space look-at point while interacting
    float duration = 4.0f;              // seconds the animation runs while the camera holds
    unsigned anims = 0;                 // AnimFlags
    // rotAxis is MODEL-space and must be a unit X, Y or Z axis (it feeds Euler animRotDeg).
    // translateBy is WORLD-space (it is added to position).
    glm::vec3 rotAxis{0.0f, 1.0f, 0.0f};
    float rotSpeedDeg = 180.0f;
    float swingAmpDeg = 15.0f, swingFreq = 1.0f;
    glm::vec3 translateBy{0.0f};
    float scaleTo = 1.3f;
    glm::vec3 colorTo{1.0f};
    glm::vec3 emissiveTo{0.0f};
    int linkedPointLight = -1;          // index into Scene::lights.points; gets +lightBoost intensity while active
    float lightBoost = 0.0f;
    std::string sound;                  // Audio::play name on trigger ("" = default "interact")
};

struct SceneObject {
    std::string name;                   // model id (assets/models/<name>.obj)
    Model* model = nullptr;
    glm::vec3 position{0.0f};
    glm::vec3 rotationDeg{0.0f};        // Euler XYZ degrees, applied Y then X then Z
    glm::vec3 scale{1.0f};
    glm::vec3 pivot{0.0f};              // model-space point that animation rotation/scale is applied about
    glm::vec3 tint{1.0f};
    glm::vec3 emissive{0.0f};
    bool solid = true;                  // blocks the player (XZ footprint of the transformed model AABB)
    HorrorAnim horror = HorrorAnim::None;
    Interaction interaction;

    // Runtime animation state; identity = initial state.
    glm::vec3 animTranslate{0.0f};
    glm::vec3 animRotDeg{0.0f};
    float animScale = 1.0f;
    glm::vec3 animTint{1.0f};
    glm::vec3 animEmissive{0.0f};

    // T(position + animTranslate) * R(rotationDeg) * S(scale) * T(pivot) * R(animRotDeg) * S(animScale) * T(-pivot)
    glm::mat4 modelMatrix() const;
    glm::vec3 worldCenter() const;      // centre of transformed AABB (ignoring animation)
    void resetAnim();
};

struct RoomShell {
    glm::vec3 size{12.0f, 4.0f, 10.0f};   // width (x), height (y), depth (z); centred on x/z, floor at y = 0
    std::string floorTex = "wood_floor";  // texture ids: assets/textures/<id>.png
    std::string wallTex = "wallpaper_damask";
    std::string ceilingTex = "plaster";
    float floorTile = 0.5f;               // texture repeats per metre
    float wallTile = 0.5f;
};

struct Scene {
    std::string name;                     // shown on HUD
    RoomShell room;
    std::vector<SceneObject> objects;
    LightRig lights;                      // lights.spot is the player's flashlight (positioned by the game)
    glm::vec3 ambient{0.06f, 0.06f, 0.08f};
    glm::vec3 fogColor{0.02f, 0.02f, 0.03f};
    float fogDensity = 0.04f;
    glm::vec3 spawnPos{0.0f};             // player feet position
    float spawnYawDeg = 0.0f;             // 0 = facing -Z, positive turns left (counter-clockwise from above)
    glm::vec2 boundsMin{-5.5f}, boundsMax{5.5f};   // walkable XZ rectangle
    std::string ambientSound = "ambient"; // looped while the scene is active
};

Scene buildLibrary(ModelCache& cache);
Scene buildBedroom(ModelCache& cache);
