#pragma once
#include <string>

// Fire-and-forget WAV playback (Windows MCI). Names map to assets/sounds/<name>.wav.
// All calls are non-blocking and silently do nothing if the file is missing or audio is unavailable.
namespace Audio {
    void init();
    void play(const std::string& name);    // one-shot; the same sound may overlap itself (small voice pool)
    void loop(const std::string& name);    // start a looping track (no-op if already looping)
    void stop(const std::string& name);    // stop a looping track or one-shot
    void shutdown();
}
