////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  convolutional
//  File:     TestHelpers.cpp
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "TestHelpers.h"

#include <iostream>


void RunTest(TestType test)
{
    try
    {
        Tensor<float,3> Y = test();
        std::cout << Y << std::endl << std::endl;
    }
    catch(std::invalid_argument e)
    {
        std::cerr << e.what() << std::endl;
    }
}