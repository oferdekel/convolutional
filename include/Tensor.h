////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     Tensor.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

// stl
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include <array>
#include <string>

template <size_t degree>
using IntTuple = std::array<size_t, degree>;

const IntTuple<3> RowMajor3TensorOrder = {2, 1, 0};
const IntTuple<4> RowMajor4TensorOrder = {3, 2, 1, 0};
const IntTuple<3> ChannelMajor3TensorOrder = {1, 0, 2};

//
// Tensor (multi-dimensional array), stored in memory in arbitrary dimension order
//
template <typename ElementType, size_t degree>
class Tensor
{
public:
    // constructors
    Tensor(IntTuple<degree> shape, IntTuple<degree> minorToMajorOrder);

    // gets the number of rows, columns, channels, ...
    size_t Size(size_t dim) const { return _shape[dim]; }

    // gets the total size of the tensor
    size_t Size() const;

    // gets a reference to a tensor element
    ElementType& operator()(IntTuple<degree> coordinate);
    const ElementType& operator()(IntTuple<degree> coordinate) const;

    // equality operator
    bool operator==(const Tensor<ElementType, degree>& other) const;
    bool operator!=(const Tensor<ElementType, degree>& other) const;

    // sets all tensor elements, other than the padding, to a given value
    void Fill(ElementType value, IntTuple<degree> padding = {});

    // runs a generator for each element in the tensor, other than the padding
    template <typename GeneratorType>
    void Generate(GeneratorType generator, IntTuple<degree> padding = {});

    // Returns a subtensor
    //Tensor<ElementType> GetSubTensor(IntTuple<degree> firstCoordinate, IntTuple<degree> shape) const;

    // Returns a pointer to the underlying contiguous data
    const ElementType* Data() const { return &_data[0]; }
    ElementType* Data() { return &_data[0]; }

    // Prints the tensor to a stream
    void Print(std::ostream& ostream) const;
    void Print(std::ostream& ostream, size_t dimension, IntTuple<degree>& index) const;

protected:
    IntTuple<degree> _shape;
    IntTuple<degree> _increments;
    std::vector<ElementType> _data;

    // increments the index
    bool Next(IntTuple<degree>& index) const;
    bool Next(IntTuple<degree>& index, IntTuple<degree> padding) const;

    // checks if a tuple is a permutation of 0, 1, 2, ...
    static bool IsOrder(IntTuple<degree> order);

    // calculates the increments from the shape and the order of the dimensions
    static IntTuple<degree> GetIncrements(IntTuple<degree> shape, IntTuple<degree> minorToMajorOrder);
};

// Streaming operator. Streams the tensor elements in logical order (row major)
template <typename ElementType, size_t degree>
std::ostream& operator<<(std::ostream& stream, const Tensor<ElementType, degree>& tensor);

template <typename ElementType, size_t degree, typename RandomEngineType>
Tensor<ElementType, degree> GetRandomTensor(RandomEngineType& engine, IntTuple<degree> shape, IntTuple<degree> minorToMajorOrder, IntTuple<degree> padding = {});

//
//
//

template <typename ElementType, size_t degree>
Tensor<ElementType, degree>::Tensor(IntTuple<degree> shape, IntTuple<degree> minorToMajorOrder) :
    _shape(shape), _increments(GetIncrements(shape, minorToMajorOrder)), _data(Size())
{}

template <typename ElementType, size_t degree>
size_t Tensor<ElementType, degree>::Size() const
{
    size_t size = _shape[0];
    for(size_t i = 1; i < degree; ++i)
    {
        size *= _shape[i];
    }
    return size;
}

template <typename ElementType, size_t degree>
ElementType& Tensor<ElementType, degree>::operator()(IntTuple<degree> coordinate)
{
    size_t index = 0;
    for (size_t i = 0; i < degree; ++i)
    {
        index += coordinate[i] * _increments[i];
    }
    return _data[index];
}

template <typename ElementType, size_t degree>
const ElementType& Tensor<ElementType, degree>::operator()(IntTuple<degree> coordinate) const
{
    size_t index = 0;
    for(size_t i = 0; i < degree; ++i)
    {
        index += coordinate[i] * _increments[i];
    }
    return _data[index];
}

template <typename ElementType, size_t degree>
bool Tensor<ElementType, degree>::operator==(const Tensor<ElementType, degree>& other) const
{
    auto elementComparer = [](ElementType a, ElementType b)
    {
        ElementType epsilon = 1.0e-5;
        return (a - b < epsilon) && (b - a < epsilon);
    };

    if (!std::equal(_shape.begin(), _shape.end(), other.Shape().begin))
    {
        return false;
    }

    IntTuple<degree> index;
    do
    {
        if ((*this)(index) != other(index))
        {
            return false;
        }
    }
    while(Next(index));

    return true;
}

template <typename ElementType, size_t degree>
bool Tensor<ElementType, degree>::operator!=(const Tensor<ElementType, degree>& other) const
{
    return !(*this == other);
}

template <typename ElementType, size_t degree>
void Tensor<ElementType, degree>::Fill(ElementType value, IntTuple<degree> padding)
{
    Generate([&](){ return value; }, padding);
}

template <typename ElementType, size_t degree>
template <typename GeneratorType>
void Tensor<ElementType, degree>::Generate(GeneratorType generator, IntTuple<degree> padding)
{
    auto index = padding;
    do
    {
        (*this)(index) = generator();
    }
    while(Next(index, padding));
}

template <typename ElementType, size_t degree>
void Tensor<ElementType, degree>::Print(std::ostream& stream) const
{
    IntTuple<degree> index = {};
    Print(stream, 0, index);
}

template <typename ElementType, size_t degree>
void Tensor<ElementType, degree>::Print(std::ostream& stream, size_t dimension, IntTuple<degree>& index) const
{
    if(dimension == degree - 1)
    {
        stream << "{ " << (*this)(index);
        for(int i = 1; i<_shape[dimension]; ++i)
        {
            Next(index);
            stream << ", " << (*this)(index);
        }
        Next(index);
        stream << " }";
    }
    else if(dimension == degree - 2)
    {
        stream << "{ ";
        Print(stream, dimension + 1, index);
        for(int i = 1; i < _shape[dimension]; ++i)
        {
            stream << ", ";
            Print(stream, dimension + 1, index);
        }
        stream << " }";
    }
    else
    {
        stream << "{ ";
        Print(stream, dimension + 1, index);
        for (int i = 1; i < _shape[dimension]; ++i)
        {
            stream << ",\n" << std::string(dimension * 2 + 2, ' ');
            Print(stream, dimension + 1, index);
        }
        stream << " }";
    }
}

template <typename ElementType, size_t degree>
bool Tensor<ElementType, degree>::Next(IntTuple<degree>& index) const
{
    size_t i = degree;
    while(i > 0)
    {
        --i;
        ++index[i];
        if(index[i] < _shape[i])
        {
            return true;
        }

        index[i] = 0;
    }
    return false;    
}

template <typename ElementType, size_t degree>
bool Tensor<ElementType, degree>::Next(IntTuple<degree>& index, IntTuple<degree> padding) const
{
    size_t i = degree;
    while(i > 0)
    {
        --i;
        ++index[i];
        if(index[i] + padding[i] < _shape[i])
        {
            return true;
        }

        index[i] = padding[i];
    }
    return false;    
}

template <typename ElementType, size_t degree>
bool Tensor<ElementType, degree>::IsOrder(IntTuple<degree> order)
{
    for(size_t i = 0; i < degree; ++i)
    {
        if (std::find(order.begin(), order.end(), i) == order.end())
        {
            return false;
        }
    }
    return true;
}

template <typename ElementType, size_t degree>
IntTuple<degree> Tensor<ElementType, degree>::GetIncrements(IntTuple<degree> shape, IntTuple<degree> minorToMajorOrder)
{
    assert(IsOrder(minorToMajorOrder));

    IntTuple<degree> increments;
    size_t size = 1;
    for (size_t i = 0; i < degree; ++i)
    {
        increments[minorToMajorOrder[i]] = size;
        size *= shape[minorToMajorOrder[i]];
    }
    return increments;
}

template <typename ElementType, size_t degree>
std::ostream& operator<<(std::ostream& stream, const Tensor<ElementType, degree>& tensor)
{
    tensor.Print(stream);
    return stream;
}

template <typename ElementType, size_t degree, typename RandomEngineType>
Tensor<ElementType, degree> GetRandomTensor(RandomEngineType& engine, IntTuple<degree> shape, IntTuple<degree> minorToMajorOrder, IntTuple<degree> padding)
{
    // create standard normal random number generator
    std::normal_distribution<ElementType> normal(0, 1);
    auto rng = [&](){ return normal(engine);};

    Tensor<ElementType, degree> T(shape, minorToMajorOrder);
    T.Generate(rng, padding);

    return T;
}
