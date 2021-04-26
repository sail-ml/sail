#pragma once

#include <sstream>
#include <stdexcept>
#include <string>


inline void MakeMessageImpl(std::ostringstream& /*os*/) {}

template <typename... Args>
void MakeMessageImpl(std::ostringstream& os, int8_t first, const Args&... args);

template <typename... Args>
void MakeMessageImpl(std::ostringstream& os, uint8_t first, const Args&... args);

template <typename Arg, typename... Args>
void MakeMessageImpl(std::ostringstream& os, const Arg& first, const Args&... args) {
    os << first;
    MakeMessageImpl(os, args...);
}

template <typename... Args>
void MakeMessageImpl(std::ostringstream& os, int8_t first, const Args&... args) {
    os << static_cast<int>(first);
    MakeMessageImpl(os, args...);
}

template <typename... Args>
void MakeMessageImpl(std::ostringstream& os, uint8_t first, const Args&... args) {
    os << static_cast<unsigned int>(first);
    MakeMessageImpl(os, args...);
}

template <typename... Args>
std::string MakeMessage(const Args&... args) {
    std::ostringstream os;
    os << std::boolalpha;
    MakeMessageImpl(os, args...);
    return os.str();
}


class SailCError : public std::runtime_error {
public:
    template <typename... Args>
    explicit SailCError(const Args&... args) : runtime_error{MakeMessage(args...)} {}
};

// Error on using invalid contexts.
class DimensionError : public SailCError {
public:
    using SailCError::SailCError;
};
// Error on using invalid contexts.
class DtypeError : public SailCError {
public:
    using SailCError::SailCError;
};

