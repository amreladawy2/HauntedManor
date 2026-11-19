#include "Shader.h"
#include "Paths.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <glm/gtc/type_ptr.hpp>

namespace {

bool readFile(const std::string& path, std::string& out) {
    std::ifstream in(resolvePath(path), std::ios::binary);
    if (!in) {
        std::cerr << "[Shader] cannot open " << path << "\n";
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    return true;
}

GLuint compile(GLenum type, const std::string& src, const char* label) {
    GLuint sh = glCreateShader(type);
    const char* p = src.c_str();
    glShaderSource(sh, 1, &p, nullptr);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetShaderiv(sh, GL_INFO_LOG_LENGTH, &len);
        std::string log(len > 0 ? len : 1, '\0');
        glGetShaderInfoLog(sh, len, nullptr, log.data());
        std::cerr << "[Shader] " << label << " compile error:\n" << log << "\n";
        glDeleteShader(sh);
        return 0;
    }
    return sh;
}

} // namespace

bool Shader::load(const std::string& vertPath, const std::string& fragPath) {
    std::string vs, fs;
    if (!readFile(vertPath, vs) || !readFile(fragPath, fs)) return false;
    return loadFromSource(vs, fs);
}

bool Shader::loadFromSource(const std::string& vertSrc, const std::string& fragSrc) {
    GLuint vs = compile(GL_VERTEX_SHADER, vertSrc, "vertex");
    GLuint fs = compile(GL_FRAGMENT_SHADER, fragSrc, "fragment");
    if (!vs || !fs) {
        glDeleteShader(vs);
        glDeleteShader(fs);
        return false;
    }

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
        std::string log(len > 0 ? len : 1, '\0');
        glGetProgramInfoLog(prog, len, nullptr, log.data());
        std::cerr << "[Shader] link error:\n" << log << "\n";
        glDeleteProgram(prog);
        return false;
    }

    if (id) glDeleteProgram(id);
    id = prog;
    cache_.clear();
    return true;
}

void Shader::use() const { glUseProgram(id); }

GLint Shader::loc(const std::string& name) const {
    auto it = cache_.find(name);
    if (it != cache_.end()) return it->second;
    GLint l = glGetUniformLocation(id, name.c_str());   // -1 (optimised out) is cached too; GL ignores it
    cache_.emplace(name, l);
    return l;
}

// glProgramUniform* so setters work whether or not this program is currently bound.
void Shader::setInt(const std::string& n, int v) const { glProgramUniform1i(id, loc(n), v); }
void Shader::setFloat(const std::string& n, float v) const { glProgramUniform1f(id, loc(n), v); }
void Shader::setVec2(const std::string& n, const glm::vec2& v) const { glProgramUniform2fv(id, loc(n), 1, glm::value_ptr(v)); }
void Shader::setVec3(const std::string& n, const glm::vec3& v) const { glProgramUniform3fv(id, loc(n), 1, glm::value_ptr(v)); }
void Shader::setVec4(const std::string& n, const glm::vec4& v) const { glProgramUniform4fv(id, loc(n), 1, glm::value_ptr(v)); }
void Shader::setMat3(const std::string& n, const glm::mat3& m) const { glProgramUniformMatrix3fv(id, loc(n), 1, GL_FALSE, glm::value_ptr(m)); }
void Shader::setMat4(const std::string& n, const glm::mat4& m) const { glProgramUniformMatrix4fv(id, loc(n), 1, GL_FALSE, glm::value_ptr(m)); }
