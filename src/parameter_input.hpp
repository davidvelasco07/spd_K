//Runtime parameter input, following the Athena++/AthenaK input file format:
//
//  <blockname>
//  name = value   # comment
//
//Values are stored as strings and converted on access. Later assignments to
//the same block/name override earlier ones (so command-line overrides simply
//re-parse "block/name=value" assignments after the file is read).
//Self-contained: no Kokkos or MPI dependencies.
#ifndef PARAMETER_INPUT_HPP_
#define PARAMETER_INPUT_HPP_

#include <map>
#include <string>
#include <vector>
#include <iostream>

class ParameterInput {
  public:
    ParameterInput() = default;

    //Read an input file; throws std::runtime_error on parse errors
    void LoadFromFile(const std::string &filename);
    //Apply a command-line override of the form "block/name=value"
    void ParseOverride(const std::string &assignment);

    bool DoesBlockExist(const std::string &block) const;
    bool DoesParameterExist(const std::string &block, const std::string &name) const;

    int         GetInteger(const std::string &block, const std::string &name) const;
    double      GetReal   (const std::string &block, const std::string &name) const;
    bool        GetBoolean(const std::string &block, const std::string &name) const;
    std::string GetString (const std::string &block, const std::string &name) const;

    //Return the stored value, or store and return the default if absent
    int         GetOrAddInteger(const std::string &block, const std::string &name, int def);
    double      GetOrAddReal   (const std::string &block, const std::string &name, double def);
    bool        GetOrAddBoolean(const std::string &block, const std::string &name, bool def);
    std::string GetOrAddString (const std::string &block, const std::string &name, const std::string &def);

    void Set(const std::string &block, const std::string &name, const std::string &value);

    //Echo the effective parameters (input-file format) for provenance
    void Dump(std::ostream &os) const;

  private:
    //block -> (name -> value); maps keep blocks/names sorted, which is fine
    std::map<std::string, std::map<std::string, std::string>> params_;
    const std::string &Fetch(const std::string &block, const std::string &name) const;
};

#endif
