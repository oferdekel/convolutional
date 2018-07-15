////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     Matrix.h
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <iostream>
#include <vector>

// Matrix Order
enum class MatrixOrder
{
    RowMajor = 0,
    ColumnMajor
};

MatrixOrder flipOrder(MatrixOrder order);

// MatrixConstInterface - imposes a matrix structure a constant external array 
template <typename ElementType>
class MatrixConstInterface
{
public:
    // constructors
    MatrixConstInterface() = default;
    MatrixConstInterface(const ElementType* data, size_t numRows, size_t numColumns, MatrixOrder order = MatrixOrder::RowMajor);

    // gets the number of rows, columns, channels
    size_t NumRows() const { return _numRows; }
    size_t NumColumns() const { return _numColumns; }
    size_t Size() const { return _numRows * _numColumns; }

    // get the order of the matrix
    MatrixOrder Order() const { return _order; }

    // gets a reference to a matrix element
    const ElementType& operator()(size_t rowIndex, size_t columnIndex) const;

    // equality operator
    //bool operator==(const MatrixConstInterface<ElementType>& other) const;
    //bool operator!=(const MatrixConstInterface<ElementType>& other) const;

    // Returns a pointer to the underlying contiguous data
    const ElementType* Data() const { return _pData; }

protected:
    size_t _numRows, _numColumns;
    size_t _rowIncrement, _columnIncrement;
    ElementType* _pData;
    MatrixOrder _order;
};

// Matrix is always streamed in the logical order (row major)
template <typename ElementType>
std::ostream& operator<<(std::ostream& stream, const MatrixConstInterface<ElementType>& matrix);

// MatrixInterface - extends MatrixConstInterface with non-const functions
template <typename ElementType>
class MatrixInterface : public MatrixConstInterface<ElementType>
{
public:
    // constructors
    using MatrixConstInterface::MatrixConstInterface;

    // gets a reference to a matrix element
    using MatrixConstInterface::operator();
    ElementType& operator()(size_t rowIndex, size_t columnIndex);

    // Returns a pointer to the underlying contiguous data
    using MatrixConstInterface::Data;
    ElementType* Data() { return _pData; }
};

// Matrix - implements the MatrixInterface without requiring external memory
template <typename ElementType>
class Matrix : public MatrixInterface<ElementType>
{
public:
    // constructors
    Matrix(size_t numRows, size_t numColumns, MatrixOrder order = MatrixOrder::RowMajor);
    Matrix(std::initializer_list<std::initializer_list<ElementType>> list, MatrixOrder order = MatrixOrder::RowMajor);

private:
    std::vector<ElementType> _data;
};

//
//
//

template <typename ElementType>
MatrixConstInterface<ElementType>::MatrixConstInterface(const ElementType* data, size_t numRows, size_t numColumns, MatrixOrder order) :
    _numRows(numRows), _numColumns(numColumns), 
    _rowIncrement(order == MatrixOrder::RowMajor ? _numColumns : 1),
    _columnIncrement(order == MatrixOrder::RowMajor ? 1 : _numRows),
    _pData(const_cast<ElementType*>(data)), _order(order)
{}

template <typename ElementType>
const ElementType& MatrixConstInterface<ElementType>::operator()(size_t rowIndex, size_t columnIndex) const
{
    return _pData[rowIndex * _rowIncrement + columnIndex * _columnIncrement];
}

template <typename ElementType>
std::ostream& operator<<(std::ostream& stream, const MatrixConstInterface<ElementType>& matrix)
{
    stream << "{ { " << matrix(0, 0);
    for (size_t j = 1; j < matrix.NumColumns(); ++j)
    {
        stream << ", " << matrix(0, j);
    }
    stream << " }";

    for (size_t i = 1; i < matrix.NumRows(); ++i)
    {
        std::cout << "," << std::endl;
        stream << "  { " << matrix(i, 0);
        for (size_t j = 1; j < matrix.NumColumns(); ++j)
        {
            stream << ", " << matrix(i, j);
        }
        stream << " }";
    }
    stream << " }";
    return stream;
}

template <typename ElementType>
ElementType& MatrixInterface<ElementType>::operator()(size_t rowIndex, size_t columnIndex)
{
    return _pData[rowIndex * _rowIncrement + columnIndex * _columnIncrement];
}

template <typename ElementType>
Matrix<ElementType>::Matrix(size_t numRows, size_t numColumns, MatrixOrder order) :
    MatrixInterface(nullptr, numRows, numColumns, order), _data(numRows * numColumns)
{
    this->_pData = _data.data();
}

template <typename ElementType>
Matrix<ElementType>::Matrix(std::initializer_list<std::initializer_list<ElementType>> list, MatrixOrder order) :
    MatrixInterface(nullptr, list.size(), list.begin()->size(), order), _data(list.size() * list.begin()->size())
{
    this->_pData = _data.data();

    size_t i = 0;
    for (auto rowIter = list.begin(); rowIter < list.end(); ++rowIter)
    {
        assert(rowIter->size() == list.begin()->size());

        size_t j = 0;
        for (auto elementIter = rowIter->begin(); elementIter < rowIter->end(); ++elementIter)
        {
            (*this)(i, j) = *elementIter;
            ++j;
        }
        ++i;
    }
}
