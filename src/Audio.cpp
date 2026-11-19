#include "Audio.h"

#ifdef _WIN32

#include "Paths.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <mmsystem.h>

#include <cstdio>
#include <deque>
#include <unordered_map>

// Every mciSendString call (open, play, even stop) blocks for 25-90 ms on current Windows builds, so all MCI
// work runs on one worker thread. The public functions only enqueue a command and return immediately.
// MCI aliases and the sound table are touched exclusively by the worker thread.

namespace {

constexpr int kVoices = 4;            // one-shot aliases per sound so footsteps etc. can overlap
constexpr size_t kMaxQueuedPlays = 16; // drop one-shots if the worker falls behind (loops/stops are never dropped)

enum class Op { Play, Loop, Stop };
struct Cmd { Op op; std::string name; };

struct Sound {
    bool tried = false;      // one-shot pool open attempted (never retried after a failure)
    std::string path;        // absolute wav path, "" if missing
    int voices = 0;          // one-shot aliases that opened successfully
    int next = 0;            // round-robin cursor
    bool loopTried = false;
    bool loopOpen = false;
    bool looping = false;
};

SRWLOCK g_lock = SRWLOCK_INIT;
CONDITION_VARIABLE g_wake = CONDITION_VARIABLE_INIT;
std::deque<Cmd> g_queue;     // guarded by g_lock
bool g_quit = false;         // guarded by g_lock
HANDLE g_thread = nullptr;   // owned by the calling (game) thread

std::unordered_map<std::string, Sound> g_sounds;  // worker thread only

bool mci(const std::string& cmd) {
    const MCIERROR e = mciSendStringA(cmd.c_str(), nullptr, 0, nullptr);
#ifdef HM_AUDIO_DEBUG  // build with -DHM_AUDIO_DEBUG to trace every MCI command and its result
    char msg[256] = "ok";
    if (e) mciGetErrorStringA(e, msg, sizeof msg);
    std::fprintf(stderr, "[Audio] %s -> %s\n", cmd.c_str(), msg);
#endif
    return e == 0;
}

// Absolute path of assets/sounds/<name>.wav, or "" if the file does not exist.
std::string wavPath(const std::string& name) {
    const std::string rel = resolvePath("assets/sounds/" + name + ".wav");
    char full[MAX_PATH];
    const DWORD n = GetFullPathNameA(rel.c_str(), MAX_PATH, full, nullptr);
    const std::string p = (n > 0 && n < MAX_PATH) ? std::string(full, n) : rel;
    return GetFileAttributesA(p.c_str()) == INVALID_FILE_ATTRIBUTES ? std::string() : p;
}

std::string voiceAlias(const std::string& name, int i) { return "hm_" + name + "_" + std::to_string(i); }
std::string loopAlias(const std::string& name) { return "hm_" + name + "_loop"; }

// MCI volume 0..1000: ambience sits under the effects.
int loopVolume(const std::string& name) { return name.rfind("ambient", 0) == 0 ? 450 : 850; }

bool openVoice(const std::string& name, Sound& s) {
    // waveaudio: the lightest MCI device for short one-shots.
    if (!mci("open \"" + s.path + "\" type waveaudio alias " + voiceAlias(name, s.voices))) return false;
    ++s.voices;
    return true;
}

void doPlay(const std::string& name) {
    Sound& s = g_sounds[name];
    if (!s.tried) {
        s.tried = true;
        s.path = wavPath(name);
        if (s.path.empty() || !openVoice(name, s)) return;
        mci("play " + voiceAlias(name, 0) + " from 0");  // sound the first use before opening the rest of the pool
        s.next = 1;
        while (s.voices < kVoices && openVoice(name, s)) {}
        s.next %= s.voices;
        return;
    }
    if (s.voices == 0) return;
    mci("play " + voiceAlias(name, s.next) + " from 0");
    s.next = (s.next + 1) % s.voices;
}

void doLoop(const std::string& name) {
    Sound& s = g_sounds[name];
    if (s.looping) return;
    if (!s.loopTried) {
        s.loopTried = true;
        const std::string p = wavPath(name);
        // mpegvideo (DirectShow) supports "repeat" and "setaudio volume"; waveaudio supports neither.
        if (p.empty() || !mci("open \"" + p + "\" type mpegvideo alias " + loopAlias(name))) return;
        s.loopOpen = true;
        mci("setaudio " + loopAlias(name) + " volume to " + std::to_string(loopVolume(name)));
    }
    if (s.loopOpen) s.looping = mci("play " + loopAlias(name) + " from 0 repeat");
}

void doStop(const std::string& name) {
    auto it = g_sounds.find(name);
    if (it == g_sounds.end()) return;
    Sound& s = it->second;
    if (s.looping) mci("stop " + loopAlias(name));
    s.looping = false;
    for (int i = 0; i < s.voices; ++i) mci("stop " + voiceAlias(name, i));
}

// Do not CoInitializeEx(COINIT_MULTITHREADED) on this thread: MCI's mpegvideo device then fails to open.
DWORD WINAPI worker(LPVOID) {
    for (;;) {
        AcquireSRWLockExclusive(&g_lock);
        while (g_queue.empty() && !g_quit) SleepConditionVariableSRW(&g_wake, &g_lock, INFINITE, 0);
        if (g_quit) {  // pending commands are discarded on shutdown
            ReleaseSRWLockExclusive(&g_lock);
            break;
        }
        Cmd c = std::move(g_queue.front());
        g_queue.pop_front();
        ReleaseSRWLockExclusive(&g_lock);

        switch (c.op) {
            case Op::Play: doPlay(c.name); break;
            case Op::Loop: doLoop(c.name); break;
            case Op::Stop: doStop(c.name); break;
        }
    }
    for (auto& [name, s] : g_sounds) {
        if (s.loopOpen) mci("close " + loopAlias(name));
        for (int i = 0; i < s.voices; ++i) mci("close " + voiceAlias(name, i));
    }
    g_sounds.clear();
    return 0;
}

void enqueue(Op op, const std::string& name) {
    if (!g_thread || name.empty()) return;
    AcquireSRWLockExclusive(&g_lock);
    if (!g_quit && (op != Op::Play || g_queue.size() < kMaxQueuedPlays)) g_queue.push_back({op, name});
    ReleaseSRWLockExclusive(&g_lock);
    WakeConditionVariable(&g_wake);
}

} // namespace

void Audio::init() {
    if (g_thread || waveOutGetNumDevs() == 0) return;  // already running, or no output device: stay silent
    g_quit = false;
    g_thread = CreateThread(nullptr, 0, worker, nullptr, 0, nullptr);  // nullptr on failure: all calls no-op
}

void Audio::play(const std::string& name) { enqueue(Op::Play, name); }
void Audio::loop(const std::string& name) { enqueue(Op::Loop, name); }
void Audio::stop(const std::string& name) { enqueue(Op::Stop, name); }

void Audio::shutdown() {
    if (!g_thread) return;
    AcquireSRWLockExclusive(&g_lock);
    g_quit = true;
    g_queue.clear();
    ReleaseSRWLockExclusive(&g_lock);
    WakeConditionVariable(&g_wake);
    // The worker finishes its current MCI call, then closes every alias; don't hang the exit if a driver stalls.
    // On timeout g_thread stays set (with g_quit) so a later init() cannot start a second worker beside it.
    if (WaitForSingleObject(g_thread, 3000) == WAIT_OBJECT_0) {
        CloseHandle(g_thread);
        g_thread = nullptr;
    }
}

#else  // no audio backend on other platforms

void Audio::init() {}
void Audio::play(const std::string&) {}
void Audio::loop(const std::string&) {}
void Audio::stop(const std::string&) {}
void Audio::shutdown() {}

#endif
