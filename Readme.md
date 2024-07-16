# SwiftTensor

A pet project: Tensor implementation written in swift, using SIMD accellecation.

The Tensor operations are well tested, both functionally and benchmarked using both conventional and Micro Benchmarks.

## Compute throughput

The SwiftTensor framework currently executes all math on the CPU. This results
in slower compute than a GPU or specialised neural engine implementation. 

A CPU / SIMD implementation has the benefit that it is easier to build and maintain.
It also may be possible to port this code to embedded devices, though that was not
the original goal of the project.

All that said, this framework targets to have the best possible performance given the tools used.
There is no memory allocation during tensor opperations, and all important functions are inlinable.
Many optimisations have been applied by profiling and analyzing the assembly output.
Benchmarks make sure that one improvement does not degrade another use case. 

## SIMD

This library employs the apple SIMD framework to boost the calculation speed.
The SIMD width is chosen statically, based on my experimentation:

Benchmark results for 784k divide(by:) operations on different SIMD<Float> sizes:

| SIMD width | divide(by:) |
|------------|-------------|
|: scalar    | 0.4488ms   :|
|: simd2     | 0.2653ms   :|
|: simd4     | 0.1316ms   :|
|: simd8     | 0.0773ms   :|
|: simd16    | 0.0706ms   :|
|: simd32    | 4.9700ms   :|

Benchmark results for 784k multiply(by:) operations:

| SIMD width | multiply(by:) |
|------------|---------------|
|: scalar    | 0.4668ms     :|
|: simd2     | 0.2588ms     :|
|: simd4     | 0.1272ms     :|
|: simd8     | 0.0802ms     :|
|: simd16    | 0.0767ms     :|
|: simd32    | 5.3037ms     :|

Benchmark results for 784k exp() operations:

| SIMD width | exp()     |
|------------|-----------|
|: scalar    | 1.7340ms :|
|: simd2     | 1.7961ms :|
|: simd4     | 1.0424ms :|
|: simd8     | 1.0113ms :|
|: simd16    | 1.0799ms :|
|: simd32    | n/a      :|

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

## Learnings

### Binary operator and the Swift optimiser

Binary operator (like `+` `*` `-` `/`) overloading can be problematic for the Swift optimiser. A binary operator must create a new instance and return it, causing an (often unneeded) memory allocation. Only when the operator implementation is inlined can the optimiser phase replace the assigned value with the allocated one.
The swift standard library marks numerical binary operators with `@_Transparent` which does multiple things:

1. When debugging, these operator implementations are hidden. The debugger steps over them always
2. The inlining is guaranteed, and takes place even before data flow validation

I'm not sure if `@_Transparent` works outside of the standard library, and it is certainly not supposed to be used outside of it.

I'll give `@_Transparent` a try, but I don't like all the edge cases that it brings.

In the mean time, binary operators on Tensors are instead implemented with `inout` static functions:

```
let a = Tensor<Shape16x16>(repeating: 2.0)
let b = Tensor<Shape16x16>(repeating: 3.0)
var result = Tensor<Shape16x16>.zero

Tensor<Shape16x16>.multiplying(into: &result, a, b)
```

This is much more ugly than `let result = a * b`. However it makes it very clear that no memory allocation
is happening during the multiplication.



