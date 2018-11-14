////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     Tensor.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>
#include <array>
#include <string>
#include <initializer_list>
#include <iomanip>
#include <cmath>

// convenient type used to store integer tuples
template <int size>
using IntTuple = std::array<int, size>;

// tensor ordering types and typical values
using TensorOrder2 = IntTuple<2>;
using TensorOrder3 = IntTuple<3>;
using TensorOrder4 = IntTuple<4>;

const TensorOrder2 RowMaj = { 1, 0 };
const TensorOrder2 ColMaj = { 0, 1 };
const TensorOrder3 RowMaj3 = { 2, 1, 0 };
const TensorOrder3 ChlMaj3 = { 1, 0, 2 };

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

    // gets the shape of the tensor
    IntTuple<degree> Shape() const { return _shape; }

    // gets a reference to a tensor element
    const ElementType& operator()(IntTuple<degree> coordinate) const;

    // equality operator
    bool operator==(const TensorConstInterface<ElementType, degree>& other) const;
    bool operator!=(const TensorConstInterface<ElementType, degree>& other) const;

// TODO
    TensorConstInterface GetSubTensor(IntTuple<degree> firstElement, IntTuple<degree> shape)
    {
        int index = 0;
        for(int i = 0; i < degree; ++i)
        {
            index += firstElement[i] * _increments[i];
        }
        ElementType* first = _pData + index;

        return TensorConstInterface(first, shape, _order, _increments);
    }

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
    bool Next(IntTuple<degree>& index, IntTuple<degree> beginPadding, IntTuple<degree> endPadding) const;

    // checks if a tuple is a permutation of 0, 1, 2, ...
    static bool IsOrder(IntTuple<degree> order);

    // calculates the increments from the shape and the order of the dimensions
    static IntTuple<degree> GetIncrements(IntTuple<degree> shape, IntTuple<degree> order);

    //  private ctor
    TensorConstInterface(const ElementType* pData, IntTuple<degree> shape, IntTuple<degree> order, IntTuple<degree> increments);
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
    void Fill(ElementType value, IntTuple<degree> beginPadding = {}, IntTuple<degree> endPadding = {});

    // runs a generator for each element in the tensor, other than the padding
    template <typename GeneratorType>
    void Generate(GeneratorType generator, IntTuple<degree> beginPadding = {}, IntTuple<degree> endPadding = {});

    // Returns a pointer to the underlying contiguous data
    using TensorConstInterface<ElementType, degree>::Data;
    ElementType* Data() { return this->_pData; }
};

//
// A Tensor implements TensorInterface but also owns its own memory
//

template <typename ElementType, int degree>
class Tensor : public TensorInterface<ElementType, degree>
{
public: 
    // constructors
    Tensor(IntTuple<degree> shape, IntTuple<degree> order);

private:
    std::vector<ElementType> _data;
};

template <typename ElementType, int degree, typename RandomEngineType>
Tensor<ElementType, degree> GetRandomTensor(RandomEngineType& engine, IntTuple<degree> shape, IntTuple<degree> order, IntTuple<degree> beginPadding = {}, IntTuple<degree> endPadding = {});

template <typename ElementType, int degree, typename RandomEngineType>
std::vector<Tensor<ElementType, degree>> GetRandomTensors(int count, RandomEngineType& engine, IntTuple<degree> shape, IntTuple<degree> order, IntTuple<degree> beginPadding = {}, IntTuple<degree> endPadding = {});

//
// Matrix (degree 2 Tensors) abbreviations
//

template <typename ElementType>
using MatrixConstInterface = TensorConstInterface<ElementType, 2>;

template <typename ElementType>
using MatrixInterface = TensorInterface<ElementType, 2>;

template <typename ElementType>
using Matrix = Tensor<ElementType, 2>;

using MatrixOrder = TensorOrder2;

//
// Matrix helper functions
//

template <typename T>
using list = std::initializer_list<T>;

template <typename ElementType>
Matrix<ElementType> GetMatrix(list<list<ElementType>> values, MatrixOrder order = RowMaj) 
{
    Matrix<ElementType> matrix({(int)values.size(), (int)values.begin()->size()}, order);
    int i = 0;
    for(auto row = values.begin(); row < values.end(); ++row)
    {
        int j = 0;
        for(auto element = row->begin(); element < row->end(); ++element)
        {
            matrix({i,j}) = *element;
            ++j;
        }
        ++i;
    }
    return matrix;
}

template <typename ElementType>
Tensor<ElementType, 3> GetTensor3(list<list<list<ElementType>>> values, TensorOrder3 order = RowMaj3) 
{
    Tensor<ElementType, 3> tensor3({(int)values.size(), (int)values.begin()->size(), (int)values.begin()->begin()->size()}, order);
    
    int i = 0;
    for(auto row = values.begin(); row < values.end(); ++row, ++i)
    {
        int j = 0;
        for(auto col = row->begin(); col < row->end(); ++col, ++j)
        {
            int k = 0;
            for(auto element = col->begin(); element < col->end(); ++element, ++k)
            {
                tensor3({i, j, k}) = *element;
            }
        }
    }
    return tensor3;
}

template <typename ElementType>
Tensor<ElementType, 4> GetTensor4(list<list<list<list<ElementType>>>> values, TensorOrder4 order = {3, 2, 1, 0}) 
{
    Tensor<ElementType, 4> tensor4(
        {(int)values.size(), (int)values.begin()->size(), (int)values.begin()->begin()->size(), (int)values.begin()->begin()->begin()->size()},
        order);
    
    int s = 0;
    for(auto t = values.begin(); t < values.end(); ++t, ++s)
    {
        int i = 0;
        for(auto row = t->begin(); row < t->end(); ++row, ++i)
        {
            int j = 0;
            for(auto col = row->begin(); col < row->end(); ++col, ++j)
            {
                int k = 0;
                for(auto element = col->begin(); element < col->end(); ++element, ++k)
                {
                    tensor4({s, i, j, k}) = *element;
                }
            }
        }
    }
    return tensor4;
}

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
        ElementType epsilon = (ElementType)1.0e-5;
        return (a - b < epsilon) && (b - a < epsilon);
    };

    if (!std::equal(_shape.begin(), _shape.end(), other.Shape().begin()))
    {
        return false;
    }

    IntTuple<degree> index = {};
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
bool TensorConstInterface<ElementType, degree>::Next(IntTuple<degree>& index, IntTuple<degree> beginPadding, IntTuple<degree> endPadding) const
{
    int i = degree;
    while(i > 0)
    {
        --i;
        ++index[i];
        if(index[i] + endPadding[i] < _shape[i])
        {
            return true;
        }

        index[i] = beginPadding[i];
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
TensorConstInterface<ElementType, degree>::TensorConstInterface(const ElementType* pData, IntTuple<degree> shape, IntTuple<degree> order, IntTuple<degree> increments) :
    _shape(shape), _increments(increments), _order(order), _pData(const_cast<ElementType*>(pData))
{}

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
void TensorInterface<ElementType, degree>::Fill(ElementType value, IntTuple<degree> beginPadding, IntTuple<degree> endPadding)
{
    Generate([&](){ return value; }, beginPadding, endPadding);
}

template <typename ElementType, int degree>
template <typename GeneratorType>
void TensorInterface<ElementType, degree>::Generate(GeneratorType generator, IntTuple<degree> beginPadding, IntTuple<degree> endPadding)
{
    auto index = beginPadding;
    do
    {
        (*this)(index) = generator();
    }
    while(this->Next(index, beginPadding, endPadding));
}

template <typename ElementType, int degree>
Tensor<ElementType, degree>::Tensor(IntTuple<degree> shape, IntTuple<degree> order) :
    TensorInterface<ElementType, degree>(nullptr, shape, order), _data(this->Size())
{
    this->_pData = _data.data();
}

template <typename ElementType, int degree, typename RandomEngineType>
Tensor<ElementType, degree> GetRandomTensor(RandomEngineType& engine, IntTuple<degree> shape, IntTuple<degree> order, IntTuple<degree> beginPadding, IntTuple<degree> endPadding)
{
    // create standard normal random number generator
    std::normal_distribution<ElementType> normal(0, 1);
    auto rng = [&](){ return normal(engine);};

    Tensor<ElementType, degree> T(shape, order);
    T.Generate(rng, beginPadding, endPadding);

    return T;
}

template <typename ElementType, int degree, typename RandomEngineType>
std::vector<Tensor<ElementType, degree>> GetRandomTensors(int count, RandomEngineType& engine, IntTuple<degree> shape, IntTuple<degree> order, IntTuple<degree> beginPadding, IntTuple<degree> endPadding)
{
    std::vector<Tensor<ElementType, degree>> tensors;
    for(int i = 0; i < count; ++i)
    {
        tensors.push_back(GetRandomTensor<ElementType, degree>(engine, shape, order, beginPadding, endPadding));
    }
    return tensors;
}

