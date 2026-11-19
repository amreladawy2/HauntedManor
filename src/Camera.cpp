#include "Camera.h"

#include "Player.h"
#include "Scene.h"

#include <algorithm>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>

namespace {

// Fraction of the segment from->to that stays inside [lo, hi] (from must be inside).
float clipToBox(const glm::vec3& from, const glm::vec3& to, const glm::vec3& lo, const glm::vec3& hi) {
    float s = 1.0f;
    for (int a = 0; a < 3; ++a) {
        const float d = to[a] - from[a];
        if (to[a] > hi[a] && d > 1e-6f) s = std::min(s, (hi[a] - from[a]) / d);
        if (to[a] < lo[a] && d < -1e-6f) s = std::min(s, (lo[a] - from[a]) / d);
    }
    return std::clamp(s, 0.0f, 1.0f);
}

// Fraction along from->to where the segment enters the box; 1 when it misses or starts inside it.
float segmentEnter(const glm::vec3& from, const glm::vec3& to, const glm::vec3& lo, const glm::vec3& hi) {
    const glm::vec3 d = to - from;
    float t0 = 0.0f, t1 = 1.0f;
    for (int a = 0; a < 3; ++a) {
        if (std::fabs(d[a]) < 1e-6f) {
            if (from[a] < lo[a] || from[a] > hi[a]) return 1.0f;
            continue;
        }
        float ta = (lo[a] - from[a]) / d[a], tb = (hi[a] - from[a]) / d[a];
        if (ta > tb) std::swap(ta, tb);
        t0 = std::max(t0, ta);
        t1 = std::min(t1, tb);
        if (t0 > t1) return 1.0f;
    }
    return t0 > 0.0f ? t0 : 1.0f;
}

}  // namespace

CameraPose lerpPose(const CameraPose& a, const CameraPose& b, float t) {
    CameraPose p;
    p.eye = glm::mix(a.eye, b.eye, t);
    p.target = glm::mix(a.target, b.target, t);
    p.up = glm::mix(a.up, b.up, t);
    p.up = glm::length(p.up) > 1e-4f ? glm::normalize(p.up) : b.up;
    return p;
}

void Camera::setMode(ViewMode m) {
    if (m == mode) return;
    if (m == ViewMode::FirstPerson || m == ViewMode::ThirdPerson) navMode = m;
    mode = m;
    transition_ = 1.4f;
}

void Camera::togglePerspective() {
    if (isFixed())
        setMode(navMode);
    else
        setMode(mode == ViewMode::FirstPerson ? ViewMode::ThirdPerson : ViewMode::FirstPerson);
}

CameraPose Camera::desiredPose(const Player& player, const RoomShell& room, float aspect,
                               const std::vector<Aabb>& obstacles) const {
    const float W = room.size.x, H = room.size.y, D = room.size.z;
    const float tanHalf = std::tan(glm::radians(fovDeg * 0.5f));
    const float asp = std::max(aspect, 0.1f);
    CameraPose p;

    switch (mode) {
    case ViewMode::FirstPerson:
        p.eye = player.eyePosition();
        p.target = p.eye + player.lookDir();
        break;

    case ViewMode::ThirdPerson: {
        // Over-the-shoulder orbit: behind, above the head and to the right; pitch swings it on a limited arc.
        const glm::vec3 right = player.right();
        const glm::vec3 head = player.position + glm::vec3(0.0f, 1.8f, 0.0f);
        const float yaw = glm::radians(player.yawDeg);
        const float arc = glm::radians(std::clamp(player.pitchDeg, -40.0f, 30.0f));
        const glm::vec3 orbitDir(-std::sin(yaw) * std::cos(arc), std::sin(arc), -std::cos(yaw) * std::cos(arc));
        glm::vec3 eye = head - orbitDir * 2.3f + glm::vec3(0.0f, 0.6f, 0.0f) + right * 0.45f;
        p.target = head + glm::vec3(0.0f, -0.25f, 0.0f) + player.lookDir() * 4.0f + right * 0.2f;

        // Spring arm: never through walls/ceiling, never inside tall furniture.
        const glm::vec3 pivot = player.position + glm::vec3(0.0f, 1.75f, 0.0f) + right * 0.3f;
        const glm::vec3 lo(-W * 0.5f + 0.2f, 0.25f, -D * 0.5f + 0.2f);
        const glm::vec3 hi(W * 0.5f - 0.2f, H - 0.2f, D * 0.5f - 0.2f);
        float s = clipToBox(pivot, eye, lo, hi);
        for (const Aabb& b : obstacles)
            s = std::min(s, segmentEnter(pivot, eye, b.min - glm::vec3(0.2f), b.max + glm::vec3(0.2f)));
        eye = pivot + (eye - pivot) * s;
        p.eye = glm::clamp(eye, lo, hi);
        break;
    }

    case ViewMode::Top: {
        // Fit the wall tops (closest to the camera) into both the vertical and horizontal FOV.
        const float fit = std::max((D * 0.5f) / tanHalf, (W * 0.5f) / (tanHalf * asp));
        p.eye = glm::vec3(0.0f, H + fit * 1.06f, 0.0f);
        p.target = glm::vec3(0.0f);
        p.up = glm::vec3(0.0f, 0.0f, -1.0f);
        break;
    }

    case ViewMode::Front:
    case ViewMode::Side: {
        // Raised 3/4 view from outside the hidden near wall (+Z for Front, +X for Side), so furniture
        // standing along that wall doesn't hide the room. The distance fits the near edge horizontally
        // and the foreshortened floor + wall height vertically.
        const bool front = mode == ViewMode::Front;
        const float across = front ? W : D, along = front ? D : W;
        const float elev = glm::radians(32.0f);
        const float fitW = (across * 0.5f) / (tanHalf * asp) + along * 0.5f * std::cos(elev);
        const float fitH = (along * std::sin(elev) + H * std::cos(elev)) * 0.5f / tanHalf;
        const float dist = std::max(fitW, fitH) * 1.08f;
        const glm::vec3 out = front ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
        p.target = glm::vec3(0.0f, H * 0.3f, 0.0f);
        p.eye = p.target + (out * std::cos(elev) + glm::vec3(0.0f, std::sin(elev), 0.0f)) * dist;
        break;
    }
    }
    return p;
}

void Camera::follow(const CameraPose& desired, float dt) {
    transition_ = std::max(0.0f, transition_ - dt);
    float rate = 8.0f;
    if (transition_ > 0.0f)
        rate = 4.5f;                                // slow glide while switching views
    else if (mode == ViewMode::FirstPerson)
        rate = 30.0f;                               // near-rigid head camera
    else if (mode == ViewMode::ThirdPerson)
        rate = 10.0f;
    pose = lerpPose(pose, desired, 1.0f - std::exp(-rate * dt));
}

glm::mat4 Camera::view() const {
    const glm::vec3 e = eye();
    glm::vec3 f = pose.target + shakeTarget - e;
    f = glm::length(f) > 1e-5f ? glm::normalize(f) : glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 up = glm::length(pose.up) > 1e-5f ? glm::normalize(pose.up) : glm::vec3(0.0f, 1.0f, 0.0f);
    if (std::fabs(glm::dot(f, up)) > 0.999f)
        up = std::fabs(f.y) > 0.9f ? glm::vec3(0.0f, 0.0f, -1.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
    if (rollDeg != 0.0f)
        up = glm::vec3(glm::rotate(glm::mat4(1.0f), glm::radians(rollDeg), f) * glm::vec4(up, 0.0f));
    return glm::lookAt(e, e + f, up);
}

glm::mat4 Camera::projection(float aspect) const {
    return glm::perspective(glm::radians(fovDeg), std::max(aspect, 0.01f), nearPlane, farPlane);
}
