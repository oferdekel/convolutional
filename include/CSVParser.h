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
// Parser exception class
//

class ParserException : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

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

// tokenizes a string with a given delimeter character
template <typename ElementType>
std::vector<ElementType> Split(const std::string& s, char delim);

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

    // checks that the header constains each of the provided keys
    bool HeaderContains(std::vector<std::string> keys) const;

    // returns the current value of a specified column
    ElementType operator[](std::string key) const { return _map.at(key); }

    // returns the current values that correspond to a given vector of keys
    std::vector<ElementType> operator[](std::vector<std::string> keys) const;

private:
    std::ifstream _stream;
    std::vector<std::string> _header;
    MapType _map;
};

//
//
//

template<typename ElementType, typename Iterator>
void Split(const std::string& s, char delim, Iterator iterator)
{
    std::stringstream stream(s);
    ElementParser<ElementType> parser;
    std::string token;
    while (std::getline(stream, token, delim))
    {
        try 
        {
            *(iterator++) = parser(token);
        }
        catch(...)
        {
            throw ParserException("token "s + token + " could not be parsed");
        }
    }
}

template <typename ElementType>
std::vector<ElementType> Split(const std::string& s, char delim)
{
    std::vector<ElementType> elements;
    Split<ElementType>(s, delim, std::back_inserter(elements));
    return elements;
}

template <typename ElementType>
CSVParser<ElementType>::CSVParser(std::string filepath) : _stream(filepath)
{
    if (!_stream.is_open())
    {
        return;
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
    const std::string whitespace = " \t\n\v\f\r";

    _map.clear();
    std::string line;
    bool success = false;

    while(!success)
    {
        if (!_stream.good())
        {
            return;
        }

        std::getline(_stream, line);

        if(line.length() == 0)
        {
            continue;
        }

        auto pos = line.find_first_not_of(whitespace);
        if(pos == std::string::npos || line[pos] == '#')
        {
            continue;
        }

        success = true;
    }

    auto elements = Split<ElementType>(line, ',');

    if (elements.size() != _header.size())
    {
        throw ParserException("expected "s + std::to_string(_header.size()) + " parameters but only found " + std::to_string(elements.size()));
    }

    for (size_t i = 0; i < _header.size(); ++i)
    {
        _map[_header[i]] = elements[i];
    }
}

template <typename ElementType>
bool CSVParser<ElementType>::HeaderContains(std::vector<std::string> keys) const
{
    auto header = GetHeader();
    std::sort(header.begin(), header.end());
    std::sort(keys.begin(), keys.end());
    return std::includes(header.begin(), header.end(), keys.begin(), keys.end());
}

template <typename ElementType>
std::vector<ElementType> CSVParser<ElementType>::operator[](std::vector<std::string> keys) const
{
    std::vector<ElementType> values;
    std::transform(keys.begin(), keys.end(), std::back_inserter(values), [&](const std::string& key) {return _map.at(key);});
    return values;
}
