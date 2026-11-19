// Development aid: `HauntedManor.exe --shots <dir>` plays a scripted tour of both rooms (every camera view,
// every interaction triggered through the real E-key eligibility test, the quest from the Library clues to the
// win screen, the jump scare, horror mode), prints PASS/FAIL checks and saves PNG screenshots.
#include "Game.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

void Game::devTour(const std::string& outDir) {
    std::filesystem::create_directories(outDir);
    const float dt = 1.0f / 60.0f;
    int shotNo = 0, failures = 0;

    auto capture = [&](const std::string& name) {
        std::vector<unsigned char> px(size_t(fbW_) * size_t(fbH_) * 3);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, fbW_, fbH_, GL_RGB, GL_UNSIGNED_BYTE, px.data());
        char file[96];
        std::snprintf(file, sizeof file, "%02d_%s.png", shotNo++, name.c_str());
        stbi_flip_vertically_on_write(1);
        stbi_write_png((std::filesystem::path(outDir) / file).string().c_str(), fbW_, fbH_, 3, px.data(), fbW_ * 3);
    };
    // Advance the game with a fixed timestep; optionally capture the last frame.
    auto run = [&](float seconds, const std::string& shot = "") {
        const int frames = std::max(1, int(seconds / dt + 0.5f));
        for (int f = 0; f < frames && !glfwWindowShouldClose(window_); ++f) {
            glfwPollEvents();
            update(dt);
            render();
            if (f == frames - 1 && !shot.empty()) capture(shot);
            glfwSwapBuffers(window_);
        }
    };
    auto press = [&](int key) { onKey(key, GLFW_PRESS, 0); };
    auto place = [&](const glm::vec3& feet, float yaw) {
        player_.reset(feet, yaw);
        collectObstacles();
        camera_.snap(camera_.desiredPose(player_, scene().room, aspect(), obstacles3d_));
    };
    auto check = [&](bool ok, const std::string& what) {
        std::printf("[tour] %s %s\n", ok ? "PASS" : "FAIL", what.c_str());
        if (!ok) ++failures;
    };
    auto find = [&](const char* name) -> int {
        for (size_t i = 0; i < scene().objects.size(); ++i)
            if (scene().objects[i].name == name) return int(i);
        return -1;
    };
    // Stand between the object and its interaction camera, inside the range, facing the object.
    auto approach = [&](int i) {
        const SceneObject& o = scene().objects[size_t(i)];
        const Interaction& it = o.interaction;
        const glm::vec3 c = o.worldCenter();
        glm::vec2 d(it.camPos.x - o.position.x, it.camPos.z - o.position.z);
        d = glm::length(d) > 1e-3f ? glm::normalize(d) : glm::vec2(0, 1);
        const glm::vec2 stand = glm::vec2(o.position.x, o.position.z) + d * std::min(it.range * 0.7f, 1.5f);
        const glm::vec2 to = glm::vec2(c.x, c.z) - stand;
        place({stand.x, 0.0f, stand.y}, glm::degrees(std::atan2(-to.x, -to.y)));
    };
    // Full interaction through the E key; returns false if E did not trigger it.
    auto interact = [&](int i, const std::string& p, bool shots) {
        const SceneObject& o = scene().objects[size_t(i)];
        const std::string name = o.name;
        const float dur = o.interaction.duration;
        approach(i);
        run(0.3f, shots ? p + "_near_" + name : "");
        press(GLFW_KEY_E);
        check(phase_ != Phase::None, p + ": E triggers " + name);
        if (phase_ == Phase::None) return false;
        run(0.9f + dur * 0.5f, shots ? p + "_interact_" + name : "");
        run(dur * 0.5f + 1.2f);
        const SceneObject& after = scene().objects[size_t(i)];
        check(phase_ == Phase::None && after.animScale == 1.0f && after.animTranslate == glm::vec3(0.0f) &&
                  after.animRotDeg == glm::vec3(0.0f) && after.animTint == glm::vec3(1.0f),
              p + ": " + name + " back to initial state");
        return true;
    };

    run(4.6f, "start");
    press(GLFW_KEY_M), run(0.3f);
    check(!mapOpen_ && toast_ == "You have no map." && toastTimer_ > 0.0f, "M before finding the map: 'You have no map.'");

    for (int s = 0; s < 2; ++s) {
        if (s == 1) run(0.1f), press(GLFW_KEY_TAB), run(1.4f);
        const std::string p = s == 0 ? "lib" : "bed";
        run(0.5f, p + "_first_person");
        press(GLFW_KEY_4), run(1.5f, p + "_third_person");
        press(GLFW_KEY_1), run(1.5f, p + "_top");
        press(GLFW_KEY_2), run(1.5f, p + "_front");
        press(GLFW_KEY_3), run(1.5f, p + "_side");
        if (s == 0) {  // the scare must read in a far fixed view too
            press(GLFW_KEY_1), run(1.5f), press(GLFW_KEY_J), run(0.5f, "scare_in_top_view"), run(1.5f);
            check(!scareActive(), "jump scare in the Top view finishes");
        }
        press(GLFW_KEY_4), run(1.0f);

        // E while facing away from everything at spawn must do nothing.
        place(scene().spawnPos, scene().spawnYawDeg + 180.0f);
        press(GLFW_KEY_E);
        check(phase_ == Phase::None, p + ": E out of range/view does nothing");

        const int mapIdx = find(Quest::kMapObject), keyIdx = find(Quest::kKeyObject);
        if (s == 0) {
            check(mapIdx >= 0, "lib_map is placed in the Library");
            if (mapIdx >= 0) {
                approach(mapIdx), run(0.2f);
                press(GLFW_KEY_E);
                check(phase_ == Phase::None && isHidden(scene().objects[size_t(mapIdx)]),
                      "map hidden and not interactable before the 4 clues");
            }
        }

        // Every ordinary interaction (in the Library these are the 4 clues).
        for (size_t i = 0; i < scene().objects.size(); ++i) {
            const SceneObject& o = scene().objects[i];
            if (!o.interaction.enabled || int(i) == mapIdx || int(i) == keyIdx) continue;
            interact(int(i), p, true);
            if (s == 0) check(quest_.objective().find("Investigate") == 0 || quest_.mapOnDesk(), "objective tracks clues");
        }

        if (s == 0 && mapIdx >= 0) {
            check(quest_.mapOnDesk() && !isHidden(scene().objects[size_t(mapIdx)]), "4 Library clues reveal the map");
            check(toast_ == "Something rustles on the desk...", "reveal toast shown");
            press(GLFW_KEY_4);  // first person, looking down at the desk
            approach(mapIdx);
            player_.pitchDeg = -32.0f;
            camera_.snap(camera_.desiredPose(player_, scene().room, aspect(), obstacles3d_));
            quest_.revealTime = 0.0f;  // replay the materialising for the screenshots
            run(0.6f, "lib_map_appearing"), run(1.8f, "lib_map_on_desk");
            interact(mapIdx, p, true);
            check(quest_.hasMap() && isHidden(scene().objects[size_t(mapIdx)]), "map taken and gone from the desk");
            check(toast_.find("Press M") != std::string::npos, "map pickup toast shown");
            press(GLFW_KEY_M), run(0.5f, "map_overlay_from_library");
            check(mapOpen_, "M opens the map once taken");
            press(GLFW_KEY_ESCAPE), run(0.1f);
            check(!mapOpen_ && !glfwWindowShouldClose(window_), "Esc closes the map without quitting");
            press(GLFW_KEY_4);  // back to third person
        }

        if (s == 1) {
            check(keyIdx >= 0, "bed_key is placed in the Nursery");
            press(GLFW_KEY_M), run(0.5f, "map_overlay_in_nursery"), press(GLFW_KEY_M);
            if (keyIdx >= 0) {
                interact(keyIdx, p, true);
                check(riddleOpen_, "finishing the key interaction opens the riddle");
                run(0.6f, "riddle");
                int wrong = 0;
                while (quest_.isCorrect(wrong)) ++wrong;
                press(GLFW_KEY_1 + wrong);
                check(!riddleOpen_ && scareActive(), "wrong answer closes the riddle and fires the jump scare");
                run(0.45f, "scare_wrong_answer");
                run(1.5f);
                check(!scareActive() && quest_.triedWrong && !isHidden(scene().objects[size_t(keyIdx)]) &&
                          quest_.objective() == "Try the key again...",
                      "scare ends, key stays, objective 'Try the key again...'");

                const auto firstOrder = quest_.order;
                interact(keyIdx, p, false);
                check(riddleOpen_ && quest_.order != firstOrder, "riddle re-opens with reshuffled answers");
                press(GLFW_KEY_ESCAPE), run(0.1f);
                check(!riddleOpen_ && !glfwWindowShouldClose(window_), "Esc puts the key down without quitting");

                interact(keyIdx, p, false);
                int right = 0;
                while (!quest_.isCorrect(right)) ++right;
                check(std::string(Quest::kChoices[quest_.order[size_t(right)]]) == "The grandfather clock",
                      "correct answer is the grandfather clock");
                press(GLFW_KEY_1 + right);
                check(winScreen_ && quest_.won() && isHidden(scene().objects[size_t(keyIdx)]),
                      "right answer: win screen, key gone");
                run(1.8f, "win_screen");
                press(GLFW_KEY_J);
                check(!scareActive(), "J ignored while the win screen shows");
                press(GLFW_KEY_ENTER), run(0.3f);
                check(!winScreen_ && quest_.objective() == "The manor is quiet... for now.",
                      "ENTER continues; objective 'The manor is quiet... for now.'");
            }
            press(GLFW_KEY_J), run(0.5f, "jump_scare_key_J");
            check(scareActive(), "J fires the jump scare");
            run(1.5f);
            check(!scareActive(), "jump scare finishes");
        }

        place(scene().spawnPos, scene().spawnYawDeg);
        press(GLFW_KEY_H), run(2.5f, p + "_horror_third_person");
        press(GLFW_KEY_4), run(1.0f, p + "_horror_first_person");
        press(GLFW_KEY_J), run(0.5f, p + "_horror_scare_first_person"), run(1.5f);
        press(GLFW_KEY_4), press(GLFW_KEY_H), run(1.5f);
    }
    press(GLFW_KEY_I), run(0.3f, "controls_help"), press(GLFW_KEY_I);
    std::printf("[tour] %d screenshots in %s, %d failure(s)\n", shotNo, outDir.c_str(), failures);
}
