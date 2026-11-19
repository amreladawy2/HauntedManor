#pragma once
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <string>
#include <unordered_map>

class Shader {
public:
    GLuint id = 0;

    // Paths are project-relative (passed through resolvePath). Prints compile/link errors to stderr.
    bool load(const std::string& vertPath, const std::string& fragPath);
    // Build from in-memory GLSL source.
    bool loadFromSource(const std::string& vertSrc, const std::string& fragSrc);
    void use() const;

    void setInt(const std::string& name, int v) const;
    void setFloat(const std::string& name, float v) const;
    void setVec2(const std::string& name, const glm::vec2& v) const;
    void setVec3(const std::string& name, const glm::vec3& v) const;
    void setVec4(const std::string& name, const glm::vec4& v) const;
    void setMat3(const std::string& name, const glm::mat3& m) const;
    void setMat4(const std::string& name, const glm::mat4& m) const;

private:
    GLint loc(const std::string& name) const;   // cached glGetUniformLocation
    mutable std::unordered_map<std::string, GLint> cache_;
};
