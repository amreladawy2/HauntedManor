#pragma once
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class Shader;

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

// Attribute layout: location 0 = position, 1 = normal, 2 = uv.
struct Mesh {
    GLuint vao = 0, vbo = 0, ebo = 0;
    GLsizei indexCount = 0;
    GLuint texture = 0;            // diffuse map (whiteTexture() when the material has none)
    glm::vec3 baseColor{1.0f};     // MTL Kd, multiplied with the texture
    glm::vec3 emissive{0.0f};      // MTL Ke

    // Binds texture to unit 0, sets uBaseColor and uMeshEmissive, issues glDrawElements.
    void draw(const Shader& s) const;
};

Mesh uploadMesh(const std::vector<Vertex>& vertices, const std::vector<unsigned>& indices, GLuint texture);

class Model {
public:
    std::vector<Mesh> meshes;
    glm::vec3 boundsMin{0.0f}, boundsMax{0.0f};   // model-space AABB over all meshes

    // Loads any Assimp-supported file (.obj/.fbx/.gltf). Texture paths are resolved relative to the model file.
    bool load(const std::string& path);
    void draw(const Shader& s) const;
};

class ModelCache {
public:
    // Loads resolvePath("assets/models/" + id + ".obj") once. Never returns nullptr:
    // on failure prints a warning and returns a 1m placeholder cube.
    Model* get(const std::string& id);

private:
    std::unordered_map<std::string, std::unique_ptr<Model>> models_;
};
