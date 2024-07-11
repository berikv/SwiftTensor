
import simd

public struct Tensor<ShapeType: Shape>: ~Copyable {
    public typealias ScalarType = Float
    public let count = ShapeType.scalarCount

    @usableFromInline
    internal var _scalars: UnsafeMutableBufferPointer<ScalarType>

    @inlinable
    public var scalars: [ScalarType] {
        _scalars.map { $0 }
    }

    public static var zero: Tensor<ShapeType> { .init(repeating: .zero) }

    @inlinable
    public init(_ scalars: some Collection<ScalarType>) {
        assert(scalars.count == ShapeType.scalarCount)
        self._scalars = UnsafeMutableBufferPointer<ScalarType>.allocate(capacity: scalars.count)
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
        Self(_scalars)
    }

    deinit {
//        _scalars.deallocate()
    }

    @inlinable
    public subscript(index: Int) -> ScalarType {
        get { _scalars[index] }
        set { _scalars[index] = newValue }
    }

    @inlinable
    public subscript(range: Range<Int>) -> Slice<some Collection<Float>> {
        get { _scalars[range] }
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
    public static func adding(into result: inout Self, _ lhs: borrowing Self, _ rhs: borrowing Self) {
        for index in result.indices {
            result[index] = lhs[index] + rhs[index]
        }
    }

    @inlinable
    public mutating func add(_ term: borrowing Self) {
        for index in _scalars.indices {
            _scalars[index] = _scalars[index] + term[index]
        }
    }

    @inlinable
    public static func subtracting(into result: inout Self, _ lhs: borrowing Self, _ rhs: borrowing Self) {
        for index in result.indices {
            result[index] = lhs[index] - rhs[index]
        }
    }

    @inlinable
    public mutating func subtract(_ term: borrowing Self) {
        for index in _scalars.indices {
            _scalars[index] = _scalars[index] - term[index]
        }
    }

    @inlinable
    public static func multiplying(into result: inout Self, _ lhs: borrowing Self, _ rhs: borrowing Self) {
        for index in result.indices {
            result[index] = lhs[index] * rhs[index]
        }
    }

    @inlinable
    public mutating func multiply(by factor: borrowing Self) {
        for index in _scalars.indices {
            _scalars[index] = _scalars[index] * factor[index]
        }
    }

    @inlinable
    public static func dividing(into result: inout Self, _ lhs: borrowing Self, _ rhs: borrowing Self) {
        for index in result.indices {
            result[index] = lhs[index] / rhs[index]
        }
    }

    @inlinable
    public mutating func divide(by divisor: borrowing Self) {
        for index in _scalars.indices {
            _scalars[index] = _scalars[index] / divisor[index]
        }
    }
}

extension Tensor {
    @inlinable
    public func sum() -> ScalarType {
        var result = ScalarType.zero

        if count >= 32 && MemoryLayout<ScalarType>.size < 2 {
            typealias SIMD = SIMD32<ScalarType>
            var sum = SIMD.zero
            let simdCount = count / SIMD.scalarCount * SIMD.scalarCount

            for i in stride(from: 0, to: simdCount, by: SIMD.scalarCount) {
                sum += SIMD(self[i..<i+SIMD.scalarCount])
            }
            result = sum.sum()

            // Handle any remaining elements
            for i in simdCount..<count {
                result += self[i]
            }
        } else if count >= 16 && MemoryLayout<ScalarType>.size < 8 {
            typealias SIMD = SIMD16<ScalarType>
            var sum = SIMD.zero
            let simdCount = count / SIMD.scalarCount * SIMD.scalarCount

            for i in stride(from: 0, to: simdCount, by: SIMD.scalarCount) {
                sum += SIMD(self[i..<i+SIMD.scalarCount])
            }
            result = sum.sum()

            // Handle any remaining elements
            for i in simdCount..<count {
                result += self[i]
            }
        } else if count >= 8 {
            typealias SIMD = SIMD8<ScalarType>
            var sum = SIMD.zero
            let simdCount = count / SIMD.scalarCount * SIMD.scalarCount

            for i in stride(from: 0, to: simdCount, by: SIMD.scalarCount) {
                sum += SIMD(self[i..<i+SIMD.scalarCount])
            }
            result = sum.sum()

            // Handle any remaining elements
            for i in simdCount..<count {
                result += self[i]
            }
        } else if count >= 4 {
            typealias SIMD = SIMD4<ScalarType>
            var sum = SIMD.zero
            let simdCount = count / SIMD.scalarCount * SIMD.scalarCount

            for i in stride(from: 0, to: simdCount, by: SIMD.scalarCount) {
                sum += SIMD(self[i..<i+SIMD.scalarCount])
            }
            result = sum.sum()

            // Handle any remaining elements
            for i in simdCount..<count {
                result += self[i]
            }
        } else {
            // Handle small tensors
            for scalar in _scalars {
                result += scalar
            }
        }

        return result
    }

    @inlinable
    public func mean() -> ScalarType {
        return sum() / ScalarType(count)
    }
}

extension Tensor {
    @inlinable
    public func max() -> ScalarType {
        var maximum = _scalars.first!
        for scalar in _scalars {
            maximum = Swift.max(scalar, maximum)
        }
        return maximum
    }
}

extension Tensor {
    @inlinable
    public mutating func exp() {
        for index in _scalars.indices {
            _scalars[index] = Darwin.exp(_scalars[index])
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
        if count >= 32 && MemoryLayout<ScalarType>.size < 2 {
            typealias SIMD = SIMD32<ScalarType>
            let simdCount = count / SIMD.scalarCount * SIMD.scalarCount
            for simdIndex in stride(from: 0, to: simdCount, by: SIMD.scalarCount) {
                let result = simd_max(SIMD(_scalars[simdIndex..<simdIndex+SIMD.scalarCount]), SIMD.zero)
                for index in 0..<SIMD.scalarCount {
                    _scalars[simdIndex + index] = result[index]
                }
            }

            // Handle any remaining elements
            for index in simdCount..<count {
                _scalars[index] = Swift.max(_scalars[index], .zero)
            }
        } else if count >= 16 && MemoryLayout<ScalarType>.size < 8 {
            typealias SIMD = SIMD16<ScalarType>
            let simdCount = count / SIMD.scalarCount * SIMD.scalarCount
            for simdIndex in stride(from: 0, to: simdCount, by: SIMD.scalarCount) {
                let result = simd_max(SIMD(_scalars[simdIndex..<simdIndex+SIMD.scalarCount]), SIMD.zero)
                for index in 0..<SIMD.scalarCount {
                    _scalars[simdIndex + index] = result[index]
                }
            }

            // Handle any remaining elements
            for index in simdCount..<count {
                _scalars[index] = Swift.max(_scalars[index], .zero)
            }
        } else if count >= 8 {
            typealias SIMD = SIMD8<ScalarType>
            let simdCount = count / SIMD.scalarCount * SIMD.scalarCount
            for simdIndex in stride(from: 0, to: simdCount, by: SIMD.scalarCount) {
                let result = simd_max(SIMD(_scalars[simdIndex..<simdIndex+SIMD.scalarCount]), SIMD.zero)
                for index in 0..<SIMD.scalarCount {
                    _scalars[simdIndex + index] = result[index]
                }
            }

            // Handle any remaining elements
            for index in simdCount..<count {
                _scalars[index] = Swift.max(_scalars[index], .zero)
            }
        } else if count >= 4 {
            typealias SIMD = SIMD4<ScalarType>
            let simdCount = count / SIMD.scalarCount * SIMD.scalarCount
            for simdIndex in stride(from: 0, to: simdCount, by: SIMD.scalarCount) {
                let result = simd_max(SIMD(_scalars[simdIndex..<simdIndex+SIMD.scalarCount]), SIMD.zero)
                for index in 0..<SIMD.scalarCount {
                    _scalars[simdIndex + index] = result[index]
                }
            }

            // Handle any remaining elements
            for index in simdCount..<count {
                _scalars[index] = Swift.max(_scalars[index], .zero)
            }
        } else {
            // Handle small tensors
            for index in _scalars.indices {
                _scalars[index] = Swift.max(_scalars[index], .zero)
            }
        }
    }
}

extension Tensor {
    @inlinable
    public mutating func softmax() {
        let maxScalar = max()
        subtract(Self(repeating: maxScalar))
        exp()
        let sum = self.sum()
        divide(by: Self(repeating: sum))

        // Handle numerical stability
        for index in _scalars.indices {
            if !_scalars[index].isFinite {
                _scalars[index] = 0
                print("Ovelflow in \(description)...")
            }
        }
    }
}

extension Tensor {
    @inlinable
    public mutating func applyL2(lambda: ScalarType) {
        for index in _scalars.indices {
            _scalars[index] -= lambda * _scalars[index]
        }
    }

    @inlinable
    public mutating func clip(to value: ScalarType) {
        for index in _scalars.indices {
            _scalars[index] = Swift.min(Swift.max(_scalars[index], -value), value)
        }
    }
}

extension Tensor /*: CustomStringConvertible*/ {
    public var description: String {
        "<Tensor \(ShapeType.self): \(scalars)>"
    }
}

