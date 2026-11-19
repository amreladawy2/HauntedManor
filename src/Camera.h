#pragma once
#include <glm/glm.hpp>
#include <vector>

class Player;
struct RoomShell;

enum class ViewMode { FirstPerson, ThirdPerson, Top, Front, Side };

// World-space axis-aligned box (used to keep the third-person camera out of furniture).
struct Aabb {
    glm::vec3 min{0.0f}, max{0.0f};
};

struct CameraPose {
    glm::vec3 eye{0.0f, 1.65f, 0.0f};
    glm::vec3 target{0.0f, 1.65f, -1.0f};
    glm::vec3 up{0.0f, 1.0f, 0.0f};
};

CameraPose lerpPose(const CameraPose& a, const CameraPose& b, float t);

class Camera {
public:
    ViewMode mode = ViewMode::FirstPerson;
    ViewMode navMode = ViewMode::FirstPerson;   // last first/third-person mode (restored from fixed views)
    float fovDeg = 60.0f, nearPlane = 0.05f, farPlane = 100.0f;

    CameraPose pose;                            // smoothed pose used for rendering
    // Horror offsets layered on top of the smoothed pose (set by the game every frame).
    glm::vec3 shakeEye{0.0f}, shakeTarget{0.0f};
    float rollDeg = 0.0f;

    bool isFixed() const { return mode == ViewMode::Top || mode == ViewMode::Front || mode == ViewMode::Side; }
    void setMode(ViewMode m);                   // starts a smooth transition
    void togglePerspective();                   // FP <-> 3P, or back to navigation from a fixed view

    // Where the camera wants to be for the current mode. Fixed views frame the whole room for 'aspect'.
    CameraPose desiredPose(const Player& player, const RoomShell& room, float aspect,
                           const std::vector<Aabb>& obstacles) const;
    void follow(const CameraPose& desired, float dt);   // exponential smoothing toward 'desired'
    void snap(const CameraPose& p) { pose = p; }

    glm::vec3 eye() const { return pose.eye + shakeEye; }
    glm::mat4 view() const;
    glm::mat4 projection(float aspect) const;

private:
    float transition_ = 0.0f;                   // > 0 while easing between modes (slower smoothing)
};
