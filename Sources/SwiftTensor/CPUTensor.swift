import Foundation

@frozen
public
struct CPUTensor<ScalarType: Scalar, ShapeType: Shape> {
    @usableFromInline
    var _scalars: [ScalarType]
}

extension CPUTensor: Tensor {

    public var scalars: [ScalarType] { _scalars }

    @inlinable
    public init(_ scalars: [ScalarType]) {
        assert(scalars.count == ShapeType.scalarCount, "Element count must equal \(ShapeType.scalarCount)")
        _scalars = scalars
    }

    @inlinable
    public init(repeating value: ScalarType) {
        _scalars = [ScalarType](repeating: value, count: ShapeType.scalarCount)
    }

    @inlinable
    public static func random(in range: Range<ScalarType>) -> Self {
        let scalars = (0..<ShapeType.scalarCount).map { _ in
            ScalarType.random(in: range)
        }
        return Self(scalars)
    }

    @inlinable
    public subscript(index: Int) -> ScalarType {
        get { _scalars[index] }
        set { _scalars[index] = newValue }
    }

    // MARK: Operators

    @inlinable
    public static func +(lhs: Self, rhs: Self) -> Self {
        let result = zip(lhs.scalars, rhs.scalars).map(+)
        return Self(result)
    }

    @inlinable
    public static func -(lhs: Self, rhs: Self) -> Self {
        let result = zip(lhs.scalars, rhs.scalars).map(-)
        return Self(result)
    }

    @inlinable
    public static func *(lhs: Self, rhs: Self) -> Self {
        let result = zip(lhs.scalars, rhs.scalars).map(*)
        return Self(result)
    }

    @inlinable
    public static func /(lhs: Self, rhs: Self) -> Self {
        let result = zip(lhs.scalars, rhs.scalars).map(/)
        return Self(result)
    }

    @inlinable
    public static func matrixMultiply<L, R>(lhs: L, rhs: R) -> Self
    where L : Tensor, R : Tensor, ScalarType == L.ScalarType, L.ScalarType == R.ScalarType
    {
        precondition(L.ShapeType.dimensionCount == 2 && R.ShapeType.dimensionCount == 2, "Both tensors must be 2-dimensional")
        precondition(Self.ShapeType.dimensionSizes[0] == L.ShapeType.dimensionSizes[0], "Left and Result dimensions must match for matrix multiplication")
        precondition(R.ShapeType.dimensionSizes[0] == L.ShapeType.dimensionSizes[1], "Inner dimensions must match for matrix multiplication")
        precondition(Self.ShapeType.dimensionSizes[1] == R.ShapeType.dimensionSizes[1], "Right and Result dimensions must match for matrix multiplication")

        let lhsRows = L.ShapeType.dimensionSizes[0]
        let lhsCols = L.ShapeType.dimensionSizes[1]
        let rhsCols = R.ShapeType.dimensionSizes[1]

        var resultScalars = [ScalarType](repeating: 0, count: lhsRows * rhsCols)

        for i in 0..<lhsRows {
            for j in 0..<rhsCols {
                var sum: ScalarType = 0
                for k in 0..<lhsCols {
                    sum += lhs.scalars[i * lhsCols + k] * rhs.scalars[k * rhsCols + j]
                }
                resultScalars[i * rhsCols + j] = sum
            }
        }

        return Self(resultScalars)
    }

    @inlinable
    public func relu() -> Self {
        let scalars = scalars.map { Swift.max($0, 0) }
        return Self(scalars)
    }

    @inlinable
    public func min() -> ScalarType {
        scalars.min()!
    }

    @inlinable
    public func max() -> ScalarType {
        scalars.max()!
    }

    @inlinable
    public func sum() -> ScalarType {
        return scalars.reduce(.zero, +)
    }

    @inlinable
    public func mean() -> ScalarType {
        return sum() / ScalarType(scalars.count)
    }
}


public protocol ExpCompatible {
    func exp() -> Self
}

extension Float: ExpCompatible {
    @inlinable
    public func exp() -> Float {
        return Darwin.exp(self)
    }
}

extension Double: ExpCompatible {
    @inlinable
    public func exp() -> Double {
        return Darwin.exp(self)
    }
}

extension CPUTensor where ScalarType: ExpCompatible {
    @inlinable
    public func exp() -> Self {
        return Self(scalars.map { $0.exp() })
    }
}

extension CPUTensor where ScalarType: ExpCompatible & FloatingPoint {
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
