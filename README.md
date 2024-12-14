# GPGPU
General Purpose Computation on Graphics Processing Unit

## Matrix inversion
#### Implementation details: 
it works for invertible matrix of order multiple of 32, otherwise the output is not equal to inverse of input matrix;

input data type is float;

migrated and optimized from an algorithm posted on https://www.mathworks.com/matlabcentral/answers/243916-trying-to-write-a-program-that-calculates-the-inverse-of-a-3x3-matrix-my-program-works-for-some-mat#answer_324850;

Timings on RTX 3060 Ti:

2784x2784 	Time = 73716[micro s]

2048x2048 	Time = 30955[micro s]

256x256 	Time = 337[micro s]

64x64 		Time = 234[micro s]
