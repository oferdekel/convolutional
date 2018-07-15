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

//
// Tensor Order
//
enum class TensorOrder
{
    RowMajor = 0,
    ChannelMajor
};

//
// Tensor (3 dimensional matrix in either row-major or channel-major memory order)
//
template <typename ElementType>
class Tensor
{
public:
    // constructors
    Tensor() = default;
    Tensor(size_t numRows, size_t numColumns, size_t numChannels, TensorOrder order = TensorOrder::RowMajor);

    // gets the number of rows, columns, channels
    size_t NumRows() const { return _numRows; }
    size_t NumColumns() const { return _numColumns; }
    size_t NumChannels() const { return _numChannels; }
    size_t Size() const { return _numRows * _numColumns * _numChannels; }

    // get the order of the tensor
    TensorOrder Order() const { return _order; }

    // gets a reference to a matrix element
    ElementType& operator()(size_t rowIndex, size_t columnIndex, size_t channelIndex);
    const ElementType& operator()(size_t rowIndex, size_t columnIndex, size_t channelIndex) const;

    // equality operator
    bool operator==(const Tensor<ElementType>& other) const;
    bool operator!=(const Tensor<ElementType>& other) const;

    // sets all tensor elements, other than the padding, to a given value
    void Fill(ElementType value, size_t verticalPadding = 0, size_t horizontalPadding=0);

    // runs a generator for each element in the tensor, other than the padding
    template <typename GeneratorType>
    void Generate(GeneratorType generator, size_t verticalPadding = 0, size_t horizontalPadding=0);

    // Returns a subtensor
    Tensor<ElementType> GetSubTensor(size_t firstRow, size_t firstColumns, size_t firstChannel, size_t numRows, size_t numColumns, size_t numChannels) const;

    // Returns a pointer to the underlying contiguous data
    const ElementType* Data() const { return &_data[0]; }
    ElementType* Data() { return &_data[0]; }

protected:
    size_t _numRows, _numColumns, _numChannels;
    size_t _rowIncrement, _columnIncrement, _channelIncrement;
    std::vector<ElementType> _data;
    TensorOrder _order;
};

// Streaming operator. Streams the tensor elements in logical order (row major)
template <typename ElementType>
std::ostream& operator<<(std::ostream& stream, const Tensor<ElementType>& tensor);

template <typename ElementType, typename RandomEngineType>
Tensor<ElementType> GetRandomTensor(RandomEngineType& engine, size_t numRows, size_t numColumns, size_t numChannels, TensorOrder order = TensorOrder::RowMajor, size_t verticalPadding = 0, size_t horizontalPadding=0);

template <typename ElementType, typename RandomEngineType>
std::vector<Tensor<ElementType>> GetRandomTensors(size_t numTensors, RandomEngineType& engine, size_t numRows, size_t numColumns, size_t numChannels, TensorOrder order = TensorOrder::RowMajor, size_t verticalPadding = 0, size_t horizontalPadding=0);

//
//
//

template <typename ElementType>
Tensor<ElementType>::Tensor(size_t numRows, size_t numColumns, size_t numChannels, TensorOrder order)
    : _numRows(numRows), _numColumns(numColumns), _numChannels(numChannels),
      _rowIncrement(order == TensorOrder::RowMajor ? _numColumns * _numChannels : _numColumns),
      _columnIncrement(order == TensorOrder::RowMajor ? _numChannels : 1),
      _channelIncrement(order == TensorOrder::RowMajor ? 1 : _numRows * _numColumns),
      _data(numRows * numColumns * numChannels),
      _order(order)
{}

template <typename ElementType>
ElementType& Tensor<ElementType>::operator()(size_t rowIndex, size_t columnIndex, size_t channelIndex)
{
    return _data[rowIndex * _rowIncrement + columnIndex * _columnIncrement + channelIndex * _channelIncrement];
}

template <typename ElementType>
const ElementType& Tensor<ElementType>::operator()(size_t rowIndex, size_t columnIndex, size_t channelIndex) const
{
    return _data[rowIndex * _rowIncrement + columnIndex * _columnIncrement + channelIndex * _channelIncrement];
}

template <typename ElementType>
bool Tensor<ElementType>::operator==(const Tensor<ElementType>& other) const
{
    auto elementComparer = [](ElementType a, ElementType b)
    {
        ElementType epsilon = 1.0e-5;
        return (a - b < epsilon) && (b - a < epsilon);
    };

    if (_numRows != other.NumRows() || _numColumns != other.NumColumns() || _numChannels != other.NumChannels())
    {
        return false;
    }

    for (size_t i = 0; i < _numRows; ++i)
    {
        for (size_t j = 0; j < _numColumns; ++j)
        {
            for (size_t k = 0; k < _numChannels; ++k)
            {
                if (!elementComparer((*this)(i, j, k), other(i, j, k))) return false;
            }
        }
    }

    return true;
}

template <typename ElementType>
bool Tensor<ElementType>::operator!=(const Tensor<ElementType>& other) const
{
    return !(*this == other);
}

template <typename ElementType>
void Tensor<ElementType>::Fill(ElementType value, size_t verticalPadding, size_t horizontalPadding)
{
    Generate([&](){ return value; }, verticalPadding, horizontalPadding);
}

template <typename ElementType>
template <typename GeneratorType>
void Tensor<ElementType>::Generate(GeneratorType generator, size_t verticalPadding, size_t horizontalPadding)
{
    for (size_t i = verticalPadding; i + verticalPadding < _numRows; ++i)
    {
        for (size_t j = horizontalPadding; j + horizontalPadding < _numColumns; ++j)
        {
            for (size_t k = 0; k < _numChannels; ++k)
            {
                (*this)(i,j,k) = generator();
            }
        }
    }
}

template <typename ElementType>
Tensor<ElementType> Tensor<ElementType>::GetSubTensor(size_t firstRow, size_t firstColumns, size_t firstChannel, size_t numRows, size_t numColumns, size_t numChannels) const
{
    return {};
}


template <typename ElementType>
std::ostream& operator<<(std::ostream& stream, const Tensor<ElementType>& tensor)
{
    stream << "{ { { " << tensor(0, 0, 0);
    for (size_t k = 1; k < tensor.NumChannels(); ++k)
    {
        stream << ", " << tensor(0, 0, k);
    }
    stream << " }";

    for (size_t j = 1; j < tensor.NumColumns(); ++j)
    {
        stream << ", { " << tensor(0, j, 0);
        for (size_t k = 1; k < tensor.NumChannels(); ++k)
        {
            stream << ", " << tensor(0, j, k);
        }
        stream << " }";
    }
    stream << " }";

    for (size_t i = 1; i < tensor.NumRows(); ++i)
    {
        stream << "," << std::endl << "  { { " << tensor(i, 0, 0);
        for (size_t k = 1; k < tensor.NumChannels(); ++k)
        {
            stream << ", " << tensor(i, 0, k);
        }
        stream << " }";

        for (size_t j = 1; j < tensor.NumColumns(); ++j)
        {
            stream << ", { " << tensor(i, j, 0);
            for (size_t k = 1; k < tensor.NumChannels(); ++k)
            {
                stream << ", " << tensor(i, j, k);
            }
            stream << " }";
        }
        stream << " }";
    }
    stream << " }";

    return stream;
}

template <typename ElementType, typename RandomEngineType>
Tensor<ElementType> GetRandomTensor(RandomEngineType& engine, size_t numRows, size_t numColumns, size_t numChannels, TensorOrder order, size_t verticalPadding, size_t horizontalPadding)
{
    // create standard normal random number generator
    std::normal_distribution<ElementType> normal(0, 1);
    auto rng = [&](){ return normal(engine);};

    Tensor<ElementType> T(numRows, numColumns, numChannels, order);
    T.Generate(rng, verticalPadding, horizontalPadding);

    return T;
}

template <typename ElementType, typename RandomEngineType>
std::vector<Tensor<ElementType>> GetRandomTensors(size_t numTensors, RandomEngineType& engine, size_t numRows, size_t numColumns, size_t numChannels, TensorOrder order, size_t verticalPadding, size_t horizontalPadding)
{
    std::vector<Tensor<ElementType>> tensors;
    for(int i=0; i<numTensors; ++i)
    {
        tensors.push_back(GetRandomTensor<ElementType>(engine, numRows, numColumns, numChannels, order, verticalPadding, horizontalPadding));
    }

    return tensors;
}