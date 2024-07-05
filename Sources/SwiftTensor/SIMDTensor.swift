import simd
import Accelerate

@frozen
public struct SIMDTensor<ScalarType: Scalar, ShapeType: Shape>
where ScalarType: SIMDScalar
{
    @usableFromInline
    typealias SIMD = SIMD8<ScalarType>

    @usableFromInline
    var _scalars: [ScalarType]

    // MARK: Internal utils

    @usableFromInline
    static var paddingCount: Int {
        let simdSize = SIMD.scalarCount
        return (simdSize - (ShapeType.scalarCount % simdSize)) % simdSize
    }

    @inlinable
    init(_unsafePadded: [ScalarType]) {
        _scalars = _unsafePadded
    }

    @inlinable
    var _simdIndices: some Sequence<Int> {
        stride(from: 0, to: _scalars.count - Self.paddingCount, by: SIMD.scalarCount)
    }

    @inlinable
    func _simd(at i: Int) -> SIMD {
        SIMD(_scalars[i..<i+SIMD.scalarCount])
    }
}

extension SIMDTensor: Tensor {
    @inlinable
    public var scalars: [ScalarType] {
        return Array(_scalars[0..<ShapeType.scalarCount])
    }

    @inlinable
    public init(_ scalars: [ScalarType]) {
        assert(scalars.count == ShapeType.scalarCount, "Element count must equal \(ShapeType.scalarCount)")
        _scalars = scalars + [ScalarType](repeating: .zero, count: Self.paddingCount)
    }

    @inlinable
    public init(repeating value: ScalarType) {
        _scalars = [ScalarType](repeating: value, count: ShapeType.scalarCount) + [ScalarType](repeating: .zero, count: Self.paddingCount)
    }

    @inlinable
    public static func random(in range: Range<ScalarType>) -> Self {
        // TODO: test if calling SIMD.random is faster
        Self(CPUTensor<ScalarType, ShapeType>.random(in: range).scalars)
    }

    @inlinable
    public subscript(index: Int) -> ScalarType {
        get { _scalars[index] }
        set { _scalars[index] = newValue }
    }
}

extension SIMDTensor where ScalarType: FixedWidthInteger {
    @inlinable
    public static func +(lhs: Self, rhs: Self) -> Self {
        var result = [ScalarType]()
        result.reserveCapacity(lhs._scalars.count)

        for i in lhs._simdIndices {
            let sumSimd = lhs._simd(at: i) &+ rhs._simd(at: i)
            sumSimd.indices.forEach { result.append(sumSimd[$0]) }
        }

        return SIMDTensor(_unsafePadded: result)
    }

    @inlinable
    public static func -(lhs: Self, rhs: Self) -> Self {
        var result = [ScalarType]()
        result.reserveCapacity(lhs._scalars.count)

        for i in lhs._simdIndices {
            let diffSimd = lhs._simd(at: i) &- rhs._simd(at: i)
            diffSimd.indices.forEach { result.append(diffSimd[$0]) }
        }

        return SIMDTensor(_unsafePadded: result)
    }

    @inlinable
    public static func *(lhs: Self, rhs: Self) -> Self {
        var result = [ScalarType]()
        result.reserveCapacity(lhs._scalars.count)

        for i in lhs._simdIndices {
            let prodSimd = lhs._simd(at: i) &* rhs._simd(at: i)
            prodSimd.indices.forEach { result.append(prodSimd[$0]) }
        }

        return SIMDTensor(_unsafePadded: result)
    }

    @inlinable
    public static func /(lhs: Self, rhs: Self) -> Self {
        var result = [ScalarType]()
        result.reserveCapacity(lhs._scalars.count)

        for i in lhs._simdIndices {
            let prodSimd = lhs._simd(at: i) / rhs._simd(at: i)
            prodSimd.indices.forEach { result.append(prodSimd[$0]) }
        }

        return SIMDTensor(_unsafePadded: result)
    }
}

extension SIMDTensor {
    @inlinable
    public func min() -> ScalarType {
        var simdIndicesIter = _simdIndices.makeIterator()
        let lastValidIndex = _scalars.count - SIMD.scalarCount

        var current = _simd(at: simdIndicesIter.next()!)
        while let nextIndex = simdIndicesIter.next(), nextIndex != lastValidIndex {
            current = simd_min(current, _simd(at: nextIndex))
        }

        let firstScalar = current[0]
        let paddingIndices = (SIMD.scalarCount - Self.paddingCount)..<SIMD.scalarCount
        if lastValidIndex == 0 {
            for index in paddingIndices {
                current[index] = firstScalar
            }
            return current.min()
        } else {
            var last = _simd(at: lastValidIndex)
            for index in paddingIndices {
                last[index] = firstScalar
            }
            return simd_min(current, last).min()
        }
    }

    @inlinable
    public func max() -> ScalarType {
        var simdIndicesIter = _simdIndices.makeIterator()
        let lastValidIndex = _scalars.count - SIMD.scalarCount

        var current = _simd(at: simdIndicesIter.next()!)
        while let nextIndex = simdIndicesIter.next(), nextIndex != lastValidIndex {
            current = simd_max(current, _simd(at: nextIndex))
        }

        let firstScalar = current[0]
        let paddingIndices = (SIMD.scalarCount - Self.paddingCount)..<SIMD.scalarCount
        if lastValidIndex == 0 {
            for index in paddingIndices {
                current[index] = firstScalar
            }
            return current.min()
        } else {
            var last = _simd(at: lastValidIndex)
            for index in paddingIndices {
                last[index] = firstScalar
            }
            return simd_max(current, last).max()
        }
    }


    //    @inlinable
    //    public func max() -> ScalarType {
    //        // Initialize the maximum value to the least possible value for the scalar type.
    //        var maxSIMD = SIMD(repeating: .min)
    //
    //        let stride = stride(from: 0, to: _scalars.count, by: SIMD.scalarCount)
    //        let lastIndex = _scalars.count - SIMD.scalarCount
    //
    //        for index in stride {
    //            if index == lastIndex {
    //                // Account for padding
    //                var lastSIMD = _simd(at: index)
    //                for j in (SIMD.scalarCount - Self.paddingCount)..<SIMD.scalarCount {
    //                    lastSIMD[j] = .min
    //                }
    //                maxSIMD = simd_max(maxSIMD, lastSIMD)
    //            } else {
    //                maxSIMD = simd_max(maxSIMD, _simd(at: index))
    //            }
    //        }
    //
    //        return maxSIMD.max()
    //    }
}

extension SIMDTensor where ScalarType: FixedWidthInteger {
    @inlinable
    public func sum() -> ScalarType {
        var simdIndicesIter = _simdIndices.makeIterator()
        var current = _simd(at: simdIndicesIter.next()!)


        while let index = simdIndicesIter.next() {
            current &+= _simd(at: index)
        }

        return current.wrappedSum()
    }

    public func mean() -> ScalarType {
        return sum() / ScalarType(scalars.count)
    }
}

extension SIMDTensor where ScalarType: FloatingPoint {
    @inlinable
    public static func +(lhs: Self, rhs: Self) -> Self {
        var result = [ScalarType]()
        result.reserveCapacity(lhs._scalars.count)

        for i in lhs._simdIndices {
            let sumSimd = lhs._simd(at: i) + rhs._simd(at: i)
            sumSimd.indices.forEach { result.append(sumSimd[$0]) }
        }

        return SIMDTensor(_unsafePadded: result)
    }

    @inlinable
    public static func -(lhs: Self, rhs: Self) -> Self {
        var result = [ScalarType]()
        result.reserveCapacity(lhs._scalars.count)

        for i in lhs._simdIndices {
            let diffSimd = lhs._simd(at: i) - rhs._simd(at: i)
            diffSimd.indices.forEach { result.append(diffSimd[$0]) }
        }

        return SIMDTensor(_unsafePadded: result)
    }

    @inlinable
    public static func *(lhs: Self, rhs: Self) -> Self {
        var result = [ScalarType]()
        result.reserveCapacity(lhs._scalars.count)

        for i in lhs._simdIndices {
            let prodSimd = lhs._simd(at: i) * rhs._simd(at: i)
            prodSimd.indices.forEach { result.append(prodSimd[$0]) }
        }

        return SIMDTensor(_unsafePadded: result)
    }

    @inlinable
    public static func /(lhs: Self, rhs: Self) -> Self {
        var result = [ScalarType]()
        result.reserveCapacity(lhs._scalars.count)

        for i in lhs._simdIndices {
            let prodSimd = lhs._simd(at: i) / rhs._simd(at: i)
            prodSimd.indices.forEach { result.append(prodSimd[$0]) }
        }

        return SIMDTensor(_unsafePadded: result)
    }

    @inlinable
    public func min() -> ScalarType {
        // Initialize the minimum value to the largest possible value for the scalar type.
        var minSIMD = SIMD(repeating: .greatestFiniteMagnitude)

        let lastIndex = _scalars.count - SIMD.scalarCount

        for index in _simdIndices {
            if index == lastIndex {
                // Account for padding
                var lastSIMD = _simd(at: index)
                for j in (SIMD.scalarCount - Self.paddingCount)..<SIMD.scalarCount {
                    lastSIMD[j] = .greatestFiniteMagnitude // no test coverage
                }
                minSIMD = simd_min(minSIMD, lastSIMD)
            } else {
                minSIMD = simd_min(minSIMD, _simd(at: index))
            }
        }

        return minSIMD.min()
    }

    @inlinable
    public func max() -> ScalarType {
        // Initialize the maximum value to the least possible value for the scalar type.
        var maxSIMD = SIMD(repeating: -.greatestFiniteMagnitude)

        let stride = stride(from: 0, to: _scalars.count, by: SIMD.scalarCount)
        let lastIndex = _scalars.count - SIMD.scalarCount

        for index in stride {
            if index == lastIndex {
                // Account for padding
                var lastSIMD = _simd(at: index)
                for j in (SIMD.scalarCount - Self.paddingCount)..<SIMD.scalarCount {
                    lastSIMD[j] = -.greatestFiniteMagnitude
                }
                maxSIMD = simd_max(maxSIMD, lastSIMD)
            } else {
                maxSIMD = simd_max(maxSIMD, _simd(at: index))
            }
        }

        return maxSIMD.max()
    }

    @inlinable
    public func sum() -> ScalarType {
        var sum = SIMD.zero
        let lastValidIndex = _scalars.count - SIMD.scalarCount

        for i in _simdIndices {
            if i == lastValidIndex {
                var lastSIMD = _simd(at: i)
                for j in (SIMD.scalarCount - Self.paddingCount)..<SIMD.scalarCount {
                    lastSIMD[j] = 0
                }
                sum += lastSIMD
            } else {
                sum += _simd(at: i)
            }
        }

        return sum.sum()
    }

//    @inlinable
    @inlinable
    public func mean() -> ScalarType {
        return sum() / ScalarType(scalars.count)
    }

    @inlinable
    public func exp() -> Self {
        var result = [ScalarType]()
        result.reserveCapacity(_scalars.count)
        for i in _simdIndices {
            let expSimd = SwiftTensor.exp(_simd(at: i))
            expSimd.indices.forEach { result.append(expSimd[$0]) }
        }
        return Self(_unsafePadded: result)
    }
}

extension SIMDTensor {
    @inlinable
    public static func matrixMultiply<L, R>(lhs: L, rhs: R) -> SIMDTensor<ScalarType, ShapeType> where L : Tensor, R : Tensor, ScalarType == L.ScalarType, L.ScalarType == R.ScalarType {
        let result = CPUTensor<ScalarType, ShapeType>.matrixMultiply(
            lhs: CPUTensor<ScalarType, L.ShapeType>(lhs.scalars),
            rhs: CPUTensor<ScalarType, R.ShapeType>(rhs.scalars))
        return Self(result.scalars)
    }
}

extension SIMDTensor where ScalarType: SIMDStepCompatible {
    @inlinable
    public func relu() -> SIMDTensor<ScalarType, ShapeType> {
        var results = [ScalarType]()
        results.reserveCapacity(_scalars.count)
        let zero = SIMD(repeating: .zero)
        for index in _simdIndices {
            let result = simd_max(_simd(at: index), zero)
            result.indices.forEach { index in results.append(result[index]) }
        }
        return Self(_unsafePadded: results)
    }
}

extension SIMDTensor where ScalarType: ExpCompatible & FloatingPoint {
    @inlinable
    public func softmax() -> Self {
        let maxScalar = max()
        let minus = self - Self(repeating: maxScalar)
        let exp = minus.exp()
        let sum = exp.sum()
        let softmax = exp / Self(repeating: sum)
        return softmax
    }
}
