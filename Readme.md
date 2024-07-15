# SwiftTensor

A pet project: Tensor implementation written in swift, using SIMD accellecation.

The Tensor operations are well tested, both functionally and benchmarked using both conventional and Micro Benchmarks.

## SIMD

This library employs the apple SIMD framework to boost the calculation speed. The SIMD width is chosen statically, based on
my experimentation:


Benchmark results for divide(by:) on different SIMD<Float> sizes:

    scalar: 0.4488ms
    simd2: 0.2653ms
    simd4: 0.1316ms
    simd8: 0.0773ms
    simd16: 0.0706ms
    simd32: 4.9700ms

Benchmark result for multiply(by:)

    scalar: 0.4668ms
    simd2: 0.2588ms
    simd4: 0.1272ms
    simd8: 0.0802ms
    simd16: 0.0767ms
    simd32: 5.3037ms

Benchmark result for exp()

    scalar: 1.7340ms
    simd2: 1.7961ms
    simd4: 1.0424ms
    simd8: 1.0113ms
    simd16: 1.0799ms

The best bit-width for SIMD types depends on the CPU used.
For M1, benchmarks here show that a bit-width of 256 is most efficient.
There is a sudden performance cliff at 32 (512bit wide) simd size.
 

## Neural network

For any serious neural network training framework, look somewhere else.
This framework aims to be so simple that any experienced coder can quickly
understand what is going on.

## Benchmarks

The Benchmarks are expected to be run on an M1 cpu, using Xcode 16 beta 2, in release builds.
If the benchmarks are run in another environment, they are expected to fail. It can
however be interesting to see how different compilers and different processors
perform.

Currently the benchmarks use clock time comparisons.

The to be benchmarked code is run 10 times. The fastest and slowest two runs are discarded, and for the remaining runs a mean value is calculated.
This value is then compared to the expected value, with a stated margin for error.

An improvoment here is to use proc_pid_rusage() to get the ri_instructions field, which encodes the number of assembly instructions executed until that point in the program. This would make the benchmark much less flaky, by being no longer dependent on the system load and the cpu capabilities.
This would only work on MacOS, so running the benchmarks would be limited to that platform. Which may be a reasonable requirement, though it is worth mentioning.
