////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Convolutions
//  File:     CSVParser.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

// stl
#include <algorithm>
#include <cctype>
#include <exception>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace std::string_literals;

//
// Element parser specializations
//

template <typename ElementType>
struct ElementParser;

template <>
struct ElementParser<int>
{
    int operator()(std::string& s) { return std::stoi(s); }
};

template <>
struct ElementParser<long>
{
    long operator()(std::string& s) { return std::stol(s); }
};

template <>
struct ElementParser<unsigned long>
{
    unsigned long operator()(std::string& s) { return std::stoul(s); }
};

template <>
struct ElementParser<float>
{
    float operator()(std::string& s) { return std::stof(s); }
};

template <>
struct ElementParser<double>
{
    double operator()(std::string& s) { return std::stod(s); }
};

template <>
struct ElementParser<std::string>
{
    std::string operator()(std::string s) { Trim(s); return s; }

private:

    // trim from left
    static inline void LTrim(std::string& s)
    {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(),
            [](int ch)
        {
            return !std::isspace(ch);
        }
        ));
    }

    // trim from right
    static inline void RTrim(std::string& s)
    {
        s.erase(std::find_if(s.rbegin(), s.rend(),
            [](int ch)
        {
            return !std::isspace(ch);
        }
        ).base(), s.end());
    }

    // trim from right
    static inline void Trim(std::string& s)
    {
        LTrim(s);
        RTrim(s);
    }
};

//
// A simple CSV line parser
//

template<typename ElementType, typename Iterator>
void Split(const std::string& s, char delim, Iterator iterator)
{
    std::stringstream stream(s);
    ElementParser<ElementType> parser;
    std::string token;
    while (std::getline(stream, token, delim))
    {
        *(iterator++) = parser(token);
    }
}

template <typename ElementType>
std::vector<ElementType> Split(const std::string& s, char delim)
{
    std::vector<ElementType> elements;
    Split<ElementType>(s, delim, std::back_inserter(elements));
    return elements;
}

//
// A simple CSV parser
//

template <typename ElementType>
class CSVParser
{
    using MapType = std::map<std::string, ElementType>;

public:
    // constructor
    CSVParser(std::string filepath);

    // Returns true if there is more content in the file
    bool IsValid() const { return _stream.good(); }

    // Proceeds to the next row.
    void Next();

    // returns a sorted version of the header
    const std::vector<std::string>& GetHeader() const { return _header; }

    // returns the current vector of elements
    const MapType& GetMap() const { return _map; }

private:
    std::ifstream _stream;
    std::vector<std::string> _header;
    MapType _map;
};

//
//
//

template <typename ElementType>
CSVParser<ElementType>::CSVParser(std::string filepath) : _stream(filepath)
{
    if (!_stream.is_open())
    {
        throw std::runtime_error("could not open "s + filepath);
    }

    // read header line
    std::string headerLine;
    std::getline(_stream, headerLine);
    _header = Split<std::string>(headerLine, ',');

    // parse the first line of elements
    Next();
}

template <typename ElementType>
void CSVParser<ElementType>::Next()
{
    _map.clear();
    std::string line;
    while (line.length() == 0 && _stream.good())
    {
        std::getline(_stream, line);
    }

    if (!_stream.good())
    {
        return;
    }

    auto elements = Split<ElementType>(line, ',');

    if (elements.size() != _header.size())
    {
        throw std::runtime_error("expected "s + std::to_string(_header.size()) + " parameters but only found " + std::to_string(elements.size()));
    }

    for (size_t i = 0; i < _header.size(); ++i)
    {
        _map[_header[i]] = elements[i];
    }
}
