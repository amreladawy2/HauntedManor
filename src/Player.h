#pragma once
#include <glm/glm.hpp>
#include <vector>

// Axis-aligned rectangle on the XZ ground plane (x = world X, y = world Z).
struct Box2 {
    glm::vec2 min{0.0f}, max{0.0f};
};

// The walking character. Position is the feet; yaw 0 faces -Z and positive yaw turns left.
class Player {
public:
    glm::vec3 position{0.0f};
    float yawDeg = 0.0f;
    float pitchDeg = 0.0f;              // view only, clamped to +/-80

    float radius = 0.25f;
    float walkSpeed = 2.6f;             // m/s
    float runFactor = 1.7f;
    float turnSpeed = 100.0f;           // deg/s
    float pitchSpeed = 70.0f;           // deg/s
    float eyeHeight = 1.65f;

    struct Input {
        float forward = 0.0f;           // +1 W, -1 S
        float strafe = 0.0f;            // +1 D, -1 A
        float turn = 0.0f;              // +1 turn left
        float look = 0.0f;              // +1 look up
        bool run = false;
    };
    struct Events {
        bool hitBoundary = false;       // tried to walk out of the walkable rectangle
        bool footstep = false;          // play a footstep sound this frame
    };

    void reset(const glm::vec3& feet, float yaw, float pitch = 0.0f);
    Events update(const Input& in, float dt, const std::vector<Box2>& solids,
                  const glm::vec2& boundsMin, const glm::vec2& boundsMax);

    glm::vec3 forward() const;          // horizontal facing (unit)
    glm::vec3 right() const;            // horizontal right (unit)
    glm::vec3 lookDir() const;          // facing including pitch (unit)
    glm::vec3 eyePosition() const;      // first-person eye with head bob
    glm::mat4 modelMatrix() const;      // player model (faces +Z in model space) with walk bob/sway

private:
    float walkPhase_ = 0.0f;            // advances 2*pi per two steps
    float moveBlend_ = 0.0f;            // 0 idle .. 1 walking, smoothed
    float stepTimer_ = 0.0f;
};
