
import simd

@frozen
public struct Tensor<ShapeType: Shape> {

    /**
     *
     * Benchmark results for divide(by:) on different SIMD<Float> sizes:
     *
     *      scalar: 0.4488ms
     *      simd2: 0.2653ms
     *      simd4: 0.1316ms
     *      simd8: 0.0773ms
     *      simd16: 0.0706ms
     *      simd32: 4.9700ms
     *
     * Benchmark result for multiply(by:)
     *
     *      scalar: 0.4668ms
     *      simd2: 0.2588ms
     *      simd4: 0.1272ms
     *      simd8: 0.0802ms
     *      simd16: 0.0767ms
     *      simd32: 5.3037ms
     *
     * Benchmark result for exp()
     *
     *      scalar: 1.7340ms
     *      simd2: 1.7961ms
     *      simd4: 1.0424ms
     *      simd8: 1.0113ms
     *      simd16: 1.0799ms
     *
     * The best bit-width for SIMD types depends on the CPU used.
     * For M1, benchmarks here show that a bit-width of 256 is most efficient.
     * There is a sudden cliff at 32 (512bit wide) simd.
     *
     */

    public typealias ScalarType = Float
    public let count = ShapeType.scalarCount

    @usableFromInline
    typealias SIMD = SIMD16<ScalarType>

    @usableFromInline
    internal static var simdSize: Int { SIMD.scalarCount }

    @usableFromInline
    internal static var paddingCount: Int {
        (simdSize - (ShapeType.scalarCount % simdSize)) % simdSize
    }

    @usableFromInline
    internal var _scalars: UnsafeMutableBufferPointer<ScalarType>

    @inlinable
    public var scalars: [ScalarType] {
        Array(_scalars[0..<_scalars.endIndex - Self.paddingCount])
    }

    @inlinable
    public static var zero: Tensor<ShapeType> { .init(repeating: .zero) }

    @inlinable
    public init(_ scalars: some Collection<ScalarType>) {
        assert(scalars.count == ShapeType.scalarCount)
        self._scalars = UnsafeMutableBufferPointer<ScalarType>.allocate(capacity: scalars.count + Self.paddingCount)
        for (index, i) in zip(self._scalars.indices, scalars.indices) {
            self._scalars[index] = scalars[i]
        }
    }

    @inlinable
    public init(repeating scalar: ScalarType) {
        self.init(Array(repeating: scalar, count: ShapeType.scalarCount))
    }

    @inlinable
    public static func random(in range: Range<ScalarType>) -> Self {
        Self((0..<ShapeType.scalarCount).map { _ in ScalarType.random(in: range) })
    }

    @inlinable
    public func copy() -> Self {
        Self(scalars)
    }


    // Need an internal final class TensorStorage to take care of dealloc
    // (see eg https://github.com/swiftlang/swift/blob/main/stdlib/public/core/ContiguousArrayBuffer.swift#L345)
    //    deinit {
    //        _scalars.deallocate()
    //    }

    @inlinable
    public subscript(index: Int) -> ScalarType {
        get { _scalars[index] }
        set { _scalars[index] = newValue }
    }

    @inlinable
    public subscript(range: Range<Int>) -> Slice<some Collection<Float>> {
        get { _scalars[range] }
        // TODO: Write a slice based range setter
//        set { scalars[range] = newValue }
    }

    public typealias Index = Int

    public let startIndex = 0
    public let endIndex = ShapeType.scalarCount
    public let indices = 0..<ShapeType.scalarCount

    @inlinable
    public func index(after i: Int) -> Int {
        i + 1
    }

    @inlinable
    public func index(before i: Int) -> Int {
        i - 1
    }
}

extension Tensor {

    @inlinable
    public static func adding(into result: inout Self, _ lhs: borrowing Self, _ rhs: /*borrowing*/ Self) {
        lhs._scalars.withMemoryRebound(to: SIMD.self) { lhsBuffer in
            rhs._scalars.withMemoryRebound(to: SIMD.self) { rhsBuffer in
                result._scalars.withMemoryRebound(to: SIMD.self) { buffer in
                    for index in buffer.indices {
                        buffer[index] = lhsBuffer[index] + rhsBuffer[index]
                    }
                }
            }
        }
    }

    @inlinable
    public mutating func add(_ term: borrowing Self) {
        term._scalars.withMemoryRebound(to: SIMD.self) { termBuffer in
            _scalars.withMemoryRebound(to: SIMD.self) { buffer in
                for index in buffer.indices {
                    buffer[index] += termBuffer[index]
                }
            }
        }
    }

    @inlinable
    public static func subtracting(into result: inout Self, _ lhs: borrowing Self, _ rhs: /*borrowing*/ Self) {
        lhs._scalars.withMemoryRebound(to: SIMD.self) { lhsBuffer in
            rhs._scalars.withMemoryRebound(to: SIMD.self) { rhsBuffer in
                result._scalars.withMemoryRebound(to: SIMD.self) { buffer in
                    for index in buffer.indices {
                        buffer[index] = lhsBuffer[index] - rhsBuffer[index]
                    }
                }
            }
        }
    }

    @inlinable
    public mutating func subtract(_ term: borrowing Self) {
        term._scalars.withMemoryRebound(to: SIMD.self) { termBuffer in
            _scalars.withMemoryRebound(to: SIMD.self) { buffer in
                for index in buffer.indices {
                    buffer[index] -= termBuffer[index]
                }
            }
        }
    }

    @inlinable
    public static func multiplying(into result: inout Self, _ lhs: borrowing Self, _ rhs: /*borrowing*/ Self) {
        lhs._scalars.withMemoryRebound(to: SIMD.self) { lhsBuffer in
            rhs._scalars.withMemoryRebound(to: SIMD.self) { rhsBuffer in
                result._scalars.withMemoryRebound(to: SIMD.self) { buffer in
                    for index in buffer.indices {
                        buffer[index] = lhsBuffer[index] * rhsBuffer[index]
                    }
                }
            }
        }
    }

    @inlinable
    public mutating func multiply(by factor: borrowing Self) {
        factor._scalars.withMemoryRebound(to: SIMD.self) { factorBuffer in
            _scalars.withMemoryRebound(to: SIMD.self) { buffer in
                for index in buffer.indices {
                    buffer[index] *= factorBuffer[index]
                }
            }
        }
    }

    @inlinable
    public static func dividing(into result: inout Self, _ lhs: borrowing Self, _ rhs: /*borrowing*/ Self) {
        lhs._scalars.withMemoryRebound(to: SIMD.self) { lhsBuffer in
            rhs._scalars.withMemoryRebound(to: SIMD.self) { rhsBuffer in
                result._scalars.withMemoryRebound(to: SIMD.self) { buffer in
                    for index in buffer.indices {
                        buffer[index] = lhsBuffer[index] / rhsBuffer[index]
                    }
                }
            }
        }
    }

    @inlinable
    public mutating func divide(by divisor: borrowing Self) {
        divisor._scalars.withMemoryRebound(to: SIMD.self) { divisorBuffer in
            _scalars.withMemoryRebound(to: SIMD.self) { buffer in
                for index in buffer.indices {
                    buffer[index] /= divisorBuffer[index]
                }
            }
        }
    }
}

extension Tensor {
    @inlinable
    public func sum() -> ScalarType {
        // scalars.reduce(0, +)

        var sum = SIMD.zero
        let simdCount = count / SIMD.scalarCount

        _scalars.withMemoryRebound(to: SIMD.self) { buffer in
            for index in 0..<simdCount {
                sum += buffer[index]
            }
        }
        
        // Add the scalars that did not fit into a simd
        var remainingSum = ScalarType.zero
        let remainingStartIndex = simdCount * SIMD.scalarCount
        for index in remainingStartIndex..<count {
            remainingSum += _scalars[index]
        }

        return sum.sum() + remainingSum
    }

    @inlinable
    public func mean() -> ScalarType {
        return sum() / ScalarType(count)
    }
}

extension Tensor {
    @inlinable
    public func max() -> ScalarType {
        // let first = scalars.first!
        // return scalars.reduce(first, Swift.max)

        var result = _scalars.first!
        let simdCount = count / SIMD.scalarCount

        if simdCount > 0 {
            result = _scalars.withMemoryRebound(to: SIMD.self) { buffer in
                var simdResult = buffer.first!
                for index in 0..<simdCount {
                    simdResult = simd_max(simdResult, buffer[index])
                }
                return simd_reduce_max(simdResult)
            }
        }

        // Add the scalars that did not fit into a simd
        let remainingStartIndex = simdCount * SIMD.scalarCount
        for index in remainingStartIndex..<count {
            result = Swift.max(result, _scalars[index])
        }

        return result
    }
}

extension Tensor {
    @inlinable
    public mutating func exp() {
        _scalars.withMemoryRebound(to: SIMD.self) { buffer in
            for index in buffer.indices {
                buffer[index] = SwiftTensor.exp(buffer[index])
            }
        }
    }
}

extension Tensor {
    @inlinable
    public static func matrixMultiplying<LHS: Shape, RHS: Shape>(into result: inout Self, _ lhs: borrowing Tensor<LHS>, _ rhs: borrowing Tensor<RHS>) {
        precondition(LHS.dimensionCount == 2, "matrixMultiply requires in and output Tensors to be 2-dimensional")
        precondition(RHS.dimensionCount == 2, "matrixMultiply requires in and output Tensors to be 2-dimensional")
        precondition(ShapeType.dimensionCount == 2, "matrixMultiply requires in and output Tensors to be 2-dimensional")
        precondition(ShapeType.dimensionSizes[0] == LHS.dimensionSizes[0], "Left and Result dimensions must match for matrix multiplication")
        precondition(RHS.dimensionSizes[0] == LHS.dimensionSizes[1], "Inner dimensions must match for matrix multiplication")
        precondition(ShapeType.dimensionSizes[1] == RHS.dimensionSizes[1], "Right and Result dimensions must match for matrix multiplication")

        let lhsRows = LHS.dimensionSizes[0]
        let lhsCols = LHS.dimensionSizes[1]
        let rhsCols = RHS.dimensionSizes[1]

        for i in 0..<lhsRows {
            for j in 0..<rhsCols {
                var sum: ScalarType = 0
                for k in 0..<lhsCols {
                    sum += lhs[i * lhsCols + k] * rhs[k * rhsCols + j]
                }
                result[i * rhsCols + j] = sum
            }
        }
    }

    @inlinable
    public static func matrixMultiplying<LHS: Shape, RHS: Shape>(into result: inout Self, transpose lhs: borrowing Tensor<LHS>, _ rhs: borrowing Tensor<RHS>) {
        precondition(LHS.dimensionCount == 2, "matrixMultiply requires in and output Tensors to be 2-dimensional")
        precondition(RHS.dimensionCount == 2, "matrixMultiply requires in and output Tensors to be 2-dimensional")
        precondition(ShapeType.dimensionCount == 2, "matrixMultiply requires in and output Tensors to be 2-dimensional")
        precondition(ShapeType.dimensionSizes[0] == LHS.dimensionSizes[1], "Left and Result dimensions must match for matrix multiplication")
        precondition(RHS.dimensionSizes[0] == LHS.dimensionSizes[0], "Inner dimensions must match for matrix multiplication")
        precondition(ShapeType.dimensionSizes[1] == RHS.dimensionSizes[1], "Right and Result dimensions must match for matrix multiplication")

        let lhsRows = LHS.dimensionSizes[1]
        let lhsCols = LHS.dimensionSizes[0]
        let rhsCols = RHS.dimensionSizes[1]

        func transposedIndex(for index: Int) -> Int {
            let row = index / lhsCols
            let col = index % lhsCols
            return col * lhsRows + row
        }

        for i in 0..<lhsRows {
            for j in 0..<rhsCols {
                var sum: ScalarType = 0
                for k in 0..<lhsCols {
                    sum += lhs[transposedIndex(for: i * lhsCols + k)] * rhs[k * rhsCols + j]
                }
                result[i * rhsCols + j] = sum
            }
        }
    }

    @inlinable
    public static func matrixMultiplying<LHS: Shape, RHS: Shape>(into result: inout Self, _ lhs: borrowing Tensor<LHS>, transpose rhs: borrowing Tensor<RHS>) {
        precondition(LHS.dimensionCount == 2, "matrixMultiply requires in and output Tensors to be 2-dimensional")
        precondition(RHS.dimensionCount == 2, "matrixMultiply requires in and output Tensors to be 2-dimensional")
        precondition(ShapeType.dimensionCount == 2, "matrixMultiply requires in and output Tensors to be 2-dimensional")
        precondition(ShapeType.dimensionSizes[0] == LHS.dimensionSizes[0], "Left and Result dimensions must match for matrix multiplication")
        precondition(RHS.dimensionSizes[1] == LHS.dimensionSizes[1], "Inner dimensions must match for matrix multiplication")
        precondition(ShapeType.dimensionSizes[1] == RHS.dimensionSizes[0], "Right and Result dimensions must match for matrix multiplication")

        let lhsRows = LHS.dimensionSizes[0]
        let lhsCols = LHS.dimensionSizes[1]
        let rhsRows = RHS.dimensionSizes[1]
        let rhsCols = RHS.dimensionSizes[0]

        func transposedIndex(for index: Int) -> Int {
            let row = index / rhsCols
            let col = index % rhsCols
            return col * rhsRows + row
        }

        for i in 0..<lhsRows {
            for j in 0..<rhsCols {
                var sum: ScalarType = 0
                for k in 0..<lhsCols {
                    sum += lhs[i * lhsCols + k] * rhs[transposedIndex(for: k * rhsCols + j)]
                }
                result[i * rhsCols + j] = sum
            }
        }
    }
}

extension Tensor {
    @inlinable
    public mutating func relu() {
        _scalars.withMemoryRebound(to: SIMD.self) { buffer in
            for index in buffer.indices {
                buffer[index] = simd_max(buffer[index], SIMD.zero)
            }
        }
    }
}

extension Tensor {
    @inlinable
    public mutating func softmax() {
        let maxScalar = max()

        // Subtract the maximum, exponentiate, and sum
        var sum: ScalarType = 0
        let simdCount = count / SIMD.scalarCount
        let remainingStartIndex = simdCount * SIMD.scalarCount

        if simdCount > 0 {
            sum = _scalars.withMemoryRebound(to: SIMD.self) { buffer in
                var simdSum = SIMD.zero
                for index in 0..<simdCount {
                    buffer[index] -= SIMD(repeating: maxScalar)
                    buffer[index] = simd.exp(buffer[index])
                    simdSum += buffer[index]
                }
                return simd_reduce_add(simdSum)
            }
        }

        // Add the scalars that did not fit into a SIMD
        for index in remainingStartIndex..<count {
            _scalars[index] = Darwin.exp(_scalars[index] - maxScalar)
            sum += _scalars[index]
        }

        // Divide by the sum
        let invSum = 1 / sum
        if simdCount > 0 {
            _scalars.withMemoryRebound(to: SIMD.self) { buffer in
                for index in 0..<simdCount {
                    buffer[index] *= SIMD(repeating: invSum)
                }
            }
        }

        // Normalize remaining scalars
        for index in remainingStartIndex..<count {
            _scalars[index] *= invSum
        }

        // Numerical stability check
        for index in indices {
            if !self[index].isFinite {
                self[index] = 0
                print("Overflow in \(description)...")
            }
        }
    }
}

extension Tensor {
    @inlinable
    public mutating func applyL2(lambda: ScalarType) {
        _scalars.withMemoryRebound(to: SIMD.self) { buffer in
            for index in buffer.indices {
                buffer[index] -= SIMD(repeating: lambda) * buffer[index]
            }
        }
    }

    @inlinable
    public mutating func clip(to value: ScalarType) {
        _scalars.withMemoryRebound(to: SIMD.self) { buffer in
            for index in buffer.indices {
                buffer[index] = simd_clamp(buffer[index], min: value, max: value)
            }
        }
    }
}

extension Tensor /*: CustomStringConvertible*/ {
    public var description: String {
        "<Tensor \(ShapeType.self): \(scalars)>"
    }
}

