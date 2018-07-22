////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     Tensor.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

// stl
#include <algorithm> // for std::find
#include <iostream>
#include <random> // for std::normal_distribution
#include <vector>
#include <array>
#include <string>

template <int degree>
using IntTuple = std::array<int, degree>;

// predefined orderings
const IntTuple<2> RowMaj2Order = { 1, 0 };
const IntTuple<2> ColMaj2Order = { 0, 1 };
const IntTuple<3> RowMaj3Order = { 2, 1, 0 };
const IntTuple<3> ChlMaj3Order = { 1, 0, 2 };
const IntTuple<4> RowMaj4Order = {3, 2, 1, 0};
const IntTuple<4> ChlMaj4Order = {2, 1, 3, 0};

//
// A Tensor is a multi-dimensional array, which can be represented in memory in different orders. A TensorConstInterface defines all of the const methods of a tensor. A TensorConstInterface does not own allocate its own memory.
//

template <typename ElementType, int degree>
class TensorConstInterface
{
public:
    // constructor
    // pData: pointer to the data array
    // shape: the number of rows, cols, chls, ...
    // order: a permuation of 0,1,2,... which specifies the order of the dimensions in memory (from minor to major)
    TensorConstInterface(const ElementType* pData, IntTuple<degree> shape, IntTuple<degree> order);

    // gets the number of rows, columns, channels, ...
    int Size(int dim) const { return _shape[dim]; }

    // gets the total size of the tensor
    int Size() const;

    // gets the minor to major order, e.g., MinorToMajor(0) returns the index of the minor dimension
    IntTuple<degree> Order() const { return _order; }

    // gets a reference to a tensor element
    const ElementType& operator()(IntTuple<degree> coordinate) const;

    // equality operator
    bool operator==(const TensorConstInterface<ElementType, degree>& other) const;
    bool operator!=(const TensorConstInterface<ElementType, degree>& other) const;

    // Returns a pointer to the underlying contiguous data
    const ElementType* Data() const { return _pData; }

    // Prints the tensor to a stream
    void Print(std::ostream& ostream) const;

protected:
    IntTuple<degree> _shape;
    IntTuple<degree> _increments;
    IntTuple<degree> _order;
    ElementType* _pData;

    // Prints the tensor to a stream
    void Print(std::ostream& ostream, int dimension, IntTuple<degree>& index) const;

    // increments the index
    bool Next(IntTuple<degree>& index) const;
    bool Next(IntTuple<degree>& index, IntTuple<degree> padding) const;

    // checks if a tuple is a permutation of 0, 1, 2, ...
    static bool IsOrder(IntTuple<degree> order);

    // calculates the increments from the shape and the order of the dimensions
    static IntTuple<degree> GetIncrements(IntTuple<degree> shape, IntTuple<degree> order);
};

// Streaming operator. Streams the tensor elements in logical order (row major)
template <typename ElementType, int degree>
std::ostream& operator<<(std::ostream& stream, const TensorConstInterface<ElementType, degree>& tensor);

//
// A TensorInterface contains all of the non-const methods of a Tensor. A TensorInterface does not allocate its own memory
//

template <typename ElementType, int degree>
class TensorInterface : public TensorConstInterface<ElementType, degree>
{
public:
    // constructor
    using TensorConstInterface<ElementType, degree>::TensorConstInterface;

    // gets a reference to a tensor element
    using TensorConstInterface<ElementType, degree>::operator();
    ElementType& operator()(IntTuple<degree> coordinate);

    // sets all tensor elements, other than the padding, to a given value
    void Fill(ElementType value, IntTuple<degree> padding = {});

    // runs a generator for each element in the tensor, other than the padding
    template <typename GeneratorType>
    void Generate(GeneratorType generator, IntTuple<degree> padding = {});

    // Returns a pointer to the underlying contiguous data
    ElementType* Data() { return this->_pData; }
};

//
// A Tensor implements TensorInterface but also owns its own memory
//

template <typename ElementType, int degree>
class Tensor : public TensorInterface<ElementType, degree>
{
public: 
    // constructor
    Tensor(IntTuple<degree> shape, IntTuple<degree> order);

private:
    std::vector<ElementType> _data;
};

template <typename ElementType, int degree, typename RandomEngineType>
Tensor<ElementType, degree> GetRandomTensor(RandomEngineType& engine, IntTuple<degree> shape, IntTuple<degree> order, IntTuple<degree> padding = {});

//
// Matrix (degree 2 Tensors) abbreviations
//

template <typename ElementType>
using MatrixConstInterface = TensorConstInterface<ElementType, 2>;

template <typename ElementType>
using MatrixInterface = TensorInterface<ElementType, 2>;

template <typename ElementType>
using Matrix = Tensor<ElementType, 2>;

using MatrixOrder = IntTuple<2>;

//
//
//

template <typename ElementType, int degree>
TensorConstInterface<ElementType, degree>::TensorConstInterface(const ElementType* pData, IntTuple<degree> shape, IntTuple<degree> order) :
    _shape(shape), _increments(GetIncrements(shape, order)), _order(order), _pData(const_cast<ElementType*>(pData))
{}

template <typename ElementType, int degree>
int TensorConstInterface<ElementType, degree>::Size() const
{
    int size = _shape[0];
    for(int i = 1; i < degree; ++i)
    {
        size *= _shape[i];
    }
    return size;
}

template <typename ElementType, int degree>
const ElementType& TensorConstInterface<ElementType, degree>::operator()(IntTuple<degree> coordinate) const
{
    int index = 0;
    for(int i = 0; i < degree; ++i)
    {
        index += coordinate[i] * _increments[i];
    }
    return _pData[index];
}

template <typename ElementType, int degree>
bool TensorConstInterface<ElementType, degree>::operator==(const TensorConstInterface<ElementType, degree>& other) const
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

template <typename ElementType, int degree>
bool TensorConstInterface<ElementType, degree>::operator!=(const TensorConstInterface<ElementType, degree>& other) const
{
    return !(*this == other);
}

template <typename ElementType, int degree>
void TensorConstInterface<ElementType, degree>::Print(std::ostream& stream) const
{
    IntTuple<degree> index = {};
    Print(stream, 0, index);
}

#include <iomanip> // TODO
#include <cmath>

template <typename ElementType, int degree>
void TensorConstInterface<ElementType, degree>::Print(std::ostream& stream, int dimension, IntTuple<degree>& index) const
{
    if(dimension == degree - 1)
    {
        stream << "{ ";
        stream << std::setw(5) << setiosflags(std::ios::fixed) << std::setprecision(2) << (*this)(index);
        int x = 5;

        for(int i = 1; i<_shape[dimension]; ++i)
        {
            Next(index);
            stream << ", " << std::setw(5) << setiosflags(std::ios::fixed) << std::setprecision(2) << (*this)(index);
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
            if(degree == 2)
            {
                stream << std::endl << "   ";
            }
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

template <typename ElementType, int degree>
bool TensorConstInterface<ElementType, degree>::Next(IntTuple<degree>& index) const
{
    int i = degree;
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

template <typename ElementType, int degree>
bool TensorConstInterface<ElementType, degree>::Next(IntTuple<degree>& index, IntTuple<degree> padding) const
{
    int i = degree;
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

template <typename ElementType, int degree>
bool TensorConstInterface<ElementType, degree>::IsOrder(IntTuple<degree> order)
{
    for(int i = 0; i < degree; ++i)
    {
        if (std::find(order.begin(), order.end(), i) == order.end())
        {
            return false;
        }
    }
    return true;
}

template <typename ElementType, int degree>
IntTuple<degree> TensorConstInterface<ElementType, degree>::GetIncrements(IntTuple<degree> shape, IntTuple<degree> order)
{
    assert(IsOrder(order));

    IntTuple<degree> increments;
    int size = 1;
    for (int i = 0; i < degree; ++i)
    {
        increments[order[i]] = size;
        size *= shape[order[i]];
    }
    return increments;
}

template <typename ElementType, int degree>
std::ostream& operator<<(std::ostream& stream, const TensorConstInterface<ElementType, degree>& tensor)
{
    tensor.Print(stream);
    return stream;
}

template <typename ElementType, int degree>
ElementType& TensorInterface<ElementType, degree>::operator()(IntTuple<degree> coordinate)
{
    int index = 0;
    for (int i = 0; i < degree; ++i)
    {
        index += coordinate[i] * this->_increments[i];
    }
    return this->_pData[index];
}

template <typename ElementType, int degree>
void TensorInterface<ElementType, degree>::Fill(ElementType value, IntTuple<degree> padding)
{
    Generate([&](){ return value; }, padding);
}

template <typename ElementType, int degree>
template <typename GeneratorType>
void TensorInterface<ElementType, degree>::Generate(GeneratorType generator, IntTuple<degree> padding)
{
    auto index = padding;
    do
    {
        (*this)(index) = generator();
    }
    while(this->Next(index, padding));
}

template <typename ElementType, int degree>
Tensor<ElementType, degree>::Tensor(IntTuple<degree> shape, IntTuple<degree> order) :
    TensorInterface<ElementType, degree>(nullptr, shape, order), _data(this->Size())
{
    this->_pData = _data.data();
}

template <typename ElementType, int degree, typename RandomEngineType>
Tensor<ElementType, degree> GetRandomTensor(RandomEngineType& engine, IntTuple<degree> shape, IntTuple<degree> order, IntTuple<degree> padding)
{
    // create standard normal random number generator
    std::normal_distribution<ElementType> normal(0, 1);
    auto rng = [&](){ return normal(engine);};

    Tensor<ElementType, degree> T(shape, order);
    T.Generate(rng, padding);

    return T;
}
