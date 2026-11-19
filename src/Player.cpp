#include "Player.h"

#include <algorithm>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>

namespace {
constexpr float kTwoPi = 6.28318530718f;
constexpr float kStepInterval = 0.45f;  // seconds between footsteps at walking speed

bool inside(const glm::vec2& p, const Box2& b, float r) {
    return p.x > b.min.x - r && p.x < b.max.x + r && p.y > b.min.y - r && p.y < b.max.y + r;
}

// A move is blocked when it enters an obstacle; moving while already inside one (bad spawn) is allowed.
bool blocked(const glm::vec2& from, const glm::vec2& to, const std::vector<Box2>& boxes, float r) {
    for (const Box2& b : boxes)
        if (inside(to, b, r) && !inside(from, b, r)) return true;
    return false;
}
}  // namespace

void Player::reset(const glm::vec3& feet, float yaw, float pitch) {
    position = feet;
    yawDeg = yaw;
    pitchDeg = pitch;
    walkPhase_ = moveBlend_ = stepTimer_ = 0.0f;
}

Player::Events Player::update(const Input& in, float dt, const std::vector<Box2>& solids,
                              const glm::vec2& boundsMin, const glm::vec2& boundsMax) {
    Events ev;

    yawDeg = std::fmod(yawDeg + in.turn * turnSpeed * dt, 360.0f);
    pitchDeg = std::clamp(pitchDeg + in.look * pitchSpeed * dt, -80.0f, 80.0f);

    const glm::vec3 f = forward(), r = right();
    glm::vec2 dir = glm::vec2(f.x, f.z) * in.forward + glm::vec2(r.x, r.z) * in.strafe;
    if (glm::length(dir) > 1.0f) dir = glm::normalize(dir);
    const float speed = walkSpeed * (in.run ? runFactor : 1.0f);
    const glm::vec2 delta = dir * speed * dt;

    // Resolve each axis separately so the player slides along obstacles.
    const glm::vec2 start(position.x, position.z);
    glm::vec2 p = start;
    const glm::vec2 nx(p.x + delta.x, p.y);
    if (!blocked(p, nx, solids, radius)) p = nx;
    const glm::vec2 nz(p.x, p.y + delta.y);
    if (!blocked(p, nz, solids, radius)) p = nz;

    const glm::vec2 clamped = glm::clamp(p, boundsMin, boundsMax);
    if (glm::length(delta) > 0.0f && glm::length(clamped - p) > 1e-6f) ev.hitBoundary = true;
    p = clamped;

    position.x = p.x;
    position.z = p.y;

    // Walk cycle driven by the distance actually covered.
    const float moved = glm::length(p - start);
    const bool moving = dt > 0.0f && moved / dt > 0.3f;
    moveBlend_ += ((moving ? 1.0f : 0.0f) - moveBlend_) * (1.0f - std::exp(-8.0f * dt));
    if (moving) {
        const float rate = moved / dt / walkSpeed;  // 1 at walking speed
        walkPhase_ = std::fmod(walkPhase_ + dt * rate * kTwoPi / (2.0f * kStepInterval), kTwoPi);
        stepTimer_ -= dt;
        if (stepTimer_ <= 0.0f) {
            ev.footstep = true;
            stepTimer_ = kStepInterval / std::max(rate, 0.5f);
        }
    } else {
        stepTimer_ = 0.12f;  // first step sounds shortly after starting to walk
    }
    return ev;
}

glm::vec3 Player::forward() const {
    const float y = glm::radians(yawDeg);
    return {-std::sin(y), 0.0f, -std::cos(y)};
}

glm::vec3 Player::right() const {
    const float y = glm::radians(yawDeg);
    return {std::cos(y), 0.0f, -std::sin(y)};
}

glm::vec3 Player::lookDir() const {
    const float y = glm::radians(yawDeg), p = glm::radians(pitchDeg);
    return {-std::sin(y) * std::cos(p), std::sin(p), -std::cos(y) * std::cos(p)};
}

glm::vec3 Player::eyePosition() const {
    const float bobY = 0.045f * (std::fabs(std::sin(walkPhase_)) - 0.5f) * moveBlend_;
    const float swayX = 0.025f * std::sin(walkPhase_) * moveBlend_;
    return position + glm::vec3(0.0f, eyeHeight + bobY, 0.0f) + right() * swayX;
}

glm::mat4 Player::modelMatrix() const {
    // Model faces +Z; rotating by yaw+180 about Y maps +Z to forward() = (-sin yaw, 0, -cos yaw).
    const float bob = 0.035f * std::fabs(std::sin(walkPhase_)) * moveBlend_;
    glm::mat4 m = glm::translate(glm::mat4(1.0f), position + glm::vec3(0.0f, bob, 0.0f));
    m = glm::rotate(m, glm::radians(yawDeg + 180.0f + 3.0f * std::sin(walkPhase_) * moveBlend_),
                    glm::vec3(0, 1, 0));
    m = glm::rotate(m, glm::radians(3.0f * moveBlend_), glm::vec3(1, 0, 0));               // lean forward
    m = glm::rotate(m, glm::radians(2.5f * std::sin(walkPhase_) * moveBlend_), glm::vec3(0, 0, 1));  // sway
    return m;
}
