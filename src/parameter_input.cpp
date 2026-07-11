#include "parameter_input.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

std::string trim(const std::string &s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

std::string strip_comment(const std::string &s) {
    size_t pos = s.find('#');
    return pos == std::string::npos ? s : s.substr(0, pos);
}

} // namespace

void ParameterInput::LoadFromFile(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("ParameterInput: cannot open file '" + filename + "'");
    std::string line, block;
    int lineno = 0;
    while (std::getline(file, line)) {
        lineno++;
        line = trim(strip_comment(line));
        if (line.empty()) continue;
        if (line.front() == '<') {
            size_t close = line.find('>');
            if (close == std::string::npos)
                throw std::runtime_error("ParameterInput: unterminated block name at line "
                                         + std::to_string(lineno));
            block = trim(line.substr(1, close - 1));
            if (block.empty())
                throw std::runtime_error("ParameterInput: empty block name at line "
                                         + std::to_string(lineno));
            continue;
        }
        size_t eq = line.find('=');
        if (eq == std::string::npos)
            throw std::runtime_error("ParameterInput: expected 'name = value' at line "
                                     + std::to_string(lineno) + ": '" + line + "'");
        if (block.empty())
            throw std::runtime_error("ParameterInput: parameter outside any <block> at line "
                                     + std::to_string(lineno));
        std::string name = trim(line.substr(0, eq));
        std::string value = trim(line.substr(eq + 1));
        if (name.empty() || value.empty())
            throw std::runtime_error("ParameterInput: empty name or value at line "
                                     + std::to_string(lineno));
        params_[block][name] = value; //last assignment wins
    }
}

void ParameterInput::ParseOverride(const std::string &assignment) {
    size_t slash = assignment.find('/');
    size_t eq = assignment.find('=');
    if (slash == std::string::npos || eq == std::string::npos || eq < slash)
        throw std::runtime_error("ParameterInput: override must be block/name=value, got '"
                                 + assignment + "'");
    std::string block = trim(assignment.substr(0, slash));
    std::string name = trim(assignment.substr(slash + 1, eq - slash - 1));
    std::string value = trim(assignment.substr(eq + 1));
    if (block.empty() || name.empty() || value.empty())
        throw std::runtime_error("ParameterInput: bad override '" + assignment + "'");
    params_[block][name] = value;
}

bool ParameterInput::DoesBlockExist(const std::string &block) const {
    return params_.count(block) > 0;
}

bool ParameterInput::DoesParameterExist(const std::string &block, const std::string &name) const {
    auto b = params_.find(block);
    return b != params_.end() && b->second.count(name) > 0;
}

const std::string &ParameterInput::Fetch(const std::string &block, const std::string &name) const {
    auto b = params_.find(block);
    if (b != params_.end()) {
        auto p = b->second.find(name);
        if (p != b->second.end()) return p->second;
    }
    throw std::runtime_error("ParameterInput: missing parameter '" + block + "/" + name + "'");
}

int ParameterInput::GetInteger(const std::string &block, const std::string &name) const {
    return std::stoi(Fetch(block, name));
}

double ParameterInput::GetReal(const std::string &block, const std::string &name) const {
    return std::stod(Fetch(block, name));
}

bool ParameterInput::GetBoolean(const std::string &block, const std::string &name) const {
    std::string v = Fetch(block, name);
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    if (v == "true" || v == "yes" || v == "on" || v == "1") return true;
    if (v == "false" || v == "no" || v == "off" || v == "0") return false;
    throw std::runtime_error("ParameterInput: '" + block + "/" + name
                             + "' is not a boolean: '" + v + "'");
}

std::string ParameterInput::GetString(const std::string &block, const std::string &name) const {
    return Fetch(block, name);
}

void ParameterInput::Set(const std::string &block, const std::string &name,
                         const std::string &value) {
    params_[block][name] = value;
}

int ParameterInput::GetOrAddInteger(const std::string &block, const std::string &name, int def) {
    if (!DoesParameterExist(block, name)) Set(block, name, std::to_string(def));
    return GetInteger(block, name);
}

double ParameterInput::GetOrAddReal(const std::string &block, const std::string &name, double def) {
    if (!DoesParameterExist(block, name)) {
        std::ostringstream ss;
        ss.precision(17);
        ss << def;
        Set(block, name, ss.str());
    }
    return GetReal(block, name);
}

bool ParameterInput::GetOrAddBoolean(const std::string &block, const std::string &name, bool def) {
    if (!DoesParameterExist(block, name)) Set(block, name, def ? "true" : "false");
    return GetBoolean(block, name);
}

std::string ParameterInput::GetOrAddString(const std::string &block, const std::string &name,
                                           const std::string &def) {
    if (!DoesParameterExist(block, name)) Set(block, name, def);
    return GetString(block, name);
}

void ParameterInput::Dump(std::ostream &os) const {
    for (const auto &b : params_) {
        os << "<" << b.first << ">\n";
        for (const auto &p : b.second)
            os << p.first << " = " << p.second << "\n";
        os << "\n";
    }
}
