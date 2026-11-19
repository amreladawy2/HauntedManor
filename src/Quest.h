#pragma once
#include <array>
#include <glm/glm.hpp>
#include <random>
#include <string>

class Hud;
struct Scene;

// Story progress (docs/CONTRACT.md "Quest extension (v2)"). Persists across scene switches.
enum class QuestStage {
    Investigate,  // finish the 4 Library interactions
    TakeMap,      // the map lies on the desk
    SearchKey,    // map taken; find the key in the Nursery
    Riddle,       // key found at least once; riddle unsolved
    Won,
};

class Quest {
public:
    static constexpr const char* kMapObject = "lib_map";
    static constexpr const char* kKeyObject = "bed_key";
    static constexpr int kClueCount = 4;
    static constexpr int kChoiceCount = 4;
    static constexpr int kCorrectChoice = 1;  // index into kChoices: "The grandfather clock"
    static const char* const kClueObjects[kClueCount];
    static const char* const kRiddle[5];
    static const char* const kChoices[kChoiceCount];

    QuestStage stage = QuestStage::Investigate;
    bool triedWrong = false;              // objective becomes "Try the key again..."
    float playTime = 0.0f;                // seconds played, frozen at the win
    float revealTime = 0.0f;              // seconds since the map appeared on the desk
    std::array<int, kChoiceCount> order{0, 1, 2, 3};  // answer button slot -> kChoices index

    void update(float dt);
    // Records a finished interaction; returns true when it completes the Library investigation.
    bool recordClue(const std::string& objectName);
    int cluesFound() const;

    bool mapOnDesk() const { return stage == QuestStage::TakeMap; }
    bool hasMap() const { return stage >= QuestStage::SearchKey; }
    bool keyAvailable() const { return stage == QuestStage::SearchKey || stage == QuestStage::Riddle; }
    bool won() const { return stage == QuestStage::Won; }
    bool isCorrect(int slot) const { return slot >= 0 && slot < kChoiceCount && order[size_t(slot)] == kCorrectChoice; }

    std::string objective() const;
    void shuffleChoices(std::mt19937& rng);  // never repeats the previous order

private:
    std::array<bool, kClueCount> clues_{};
};

// --- Parchment HUD panels (drawn between Hud::begin and Hud::end; 'ui' = resolution scale, 1 at 900p).

// Top-down plan of the Nursery: outline, door/window marks, labelled footprints, pulsing X at the key
// (when keyStillThere) and the player's position + facing (when showPlayer). The sheet is centred in
// 'region' (x, y, w, h in framebuffer pixels) so it stays clear of the other HUD panels.
void drawNurseryMap(Hud& hud, const Scene& nursery, bool keyStillThere, bool showPlayer, const glm::vec3& playerPos,
                    float playerYawDeg, float time, float ui, const glm::vec4& region);

// Riddle modal with four answer buttons. Returns the clicked button slot (0..3) or -1.
int drawRiddle(Hud& hud, const Quest& quest, float openTime, float ui, int fbW, int fbH, double mouseX, double mouseY,
               bool clicked);

// Victory screen ("YOU ESCAPED THE MANOR", time taken, ENTER hint).
void drawWinScreen(Hud& hud, const Quest& quest, float shownTime, float ui, int fbW, int fbH);
