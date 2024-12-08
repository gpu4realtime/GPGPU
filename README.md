# GPGPU
General Purpose Computation on Graphics Processing Unit

## Matrix inversion
#### Implementation details: 
it works for invertible matrix of order multiple of 32, otherwise the output is not equal to inverse of input matrix;

input data type is float;

migrated and optimized from an algorithm posted on https://www.mathworks.com/matlabcentral/answers/243916-trying-to-write-a-program-that-calculates-the-inverse-of-a-3x3-matrix-my-program-works-for-some-mat#answer_324850;