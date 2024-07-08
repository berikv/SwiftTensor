
public protocol Tensor {
    associatedtype ScalarType: Scalar
    associatedtype ShapeType: Shape
    associatedtype C: Collection<ScalarType>

    static var zero: Self { get }
    var scalars: C { get }

    // MARK: Initialisation

    init(_ scalars: [ScalarType])
    init(repeating scalar: ScalarType)
    static func random(in range: Range<ScalarType>) -> Self

    subscript(index: Int) -> ScalarType { get set }

    func transposed() -> TransposedTensor<Self>
}

extension Tensor {
    public static var zero: Self { Self(repeating: .zero) }

    @inlinable
    public init(repeating value: ScalarType) {
        let scalars = [ScalarType](repeating: value, count: ShapeType.scalarCount)
        self.init(scalars)
    }

    @inlinable
    public static func random(in range: Range<ScalarType>) -> Self {
        let scalars = (0..<ShapeType.scalarCount).map { _ in
            ScalarType.random(in: range)
        }
        return Self(scalars)
    }

    @inlinable
    public var scalars: [ScalarType] {
        (0..<ShapeType.scalarCount).map { self[$0] }
    }

    @inlinable
    public func transposed() -> TransposedTensor<Self> {
        TransposedTensor(self)
    }

    @inlinable
    public func transpose(into result: inout TransposedTensor<Self>) {
        let originalStrides = ShapeType.computeStrides()
        let transposedStrides = ShapeDual<ShapeType>.computeStrides()

        for index in 0..<ShapeType.scalarCount {
            let transposedIndex = TransposedTensor<Self>.translateIndex(index, originalStrides: originalStrides, transposedStrides: transposedStrides)
            result.scalars[transposedIndex] = scalars[index]
        }
    }
}

extension Shape {
    @inlinable
    public static func computeStrides() -> [Int] {
        var strides = Array(repeating: 1, count: dimensionCount)
        for i in (0..<dimensionCount-1).reversed() {
            strides[i] = strides[i+1] * dimensionSizes[i+1]
        }
        return strides
    }
}

public struct TransposedTensor<T: Tensor>: Tensor {
    public typealias ScalarType = T.ScalarType
    public typealias ShapeType = ShapeDual<T.ShapeType>

    public var scalars: [ScalarType]

    @inlinable
    init(_ tensor: T) {
        var transposedScalars = Array(repeating: tensor.scalars[0], count: T.ShapeType.scalarCount)
        let originalStrides = T.ShapeType.computeStrides()
        let transposedStrides = ShapeType.computeStrides()

        for index in 0..<tensor.scalars.count {
            let transposedIndex = TransposedTensor<T>.translateIndex(index, originalStrides: originalStrides, transposedStrides: transposedStrides)
            transposedScalars[transposedIndex] = tensor.scalars[index]
        }

        self.scalars = transposedScalars
    }

    @inlinable
    public init(_ scalars: [T.ScalarType]) {
        self = Self(T(scalars))
    }

    @inlinable
    public static func random(in range: Range<T.ScalarType>) -> TransposedTensor<T> {
        Self(T.random(in: range))
    }

    @inlinable
    public subscript(index: Int) -> T.ScalarType {
        get { scalars[index] }
        set { scalars[index] = newValue }
    }

    @usableFromInline
    static func translateIndex(_ index: Int, originalStrides: [Int], transposedStrides: [Int]) -> Int {
        let transposedIndices = unravelIndex(index, shape: ShapeType.dimensionSizes, strides: transposedStrides)
        let originalIndices = Array(transposedIndices.reversed())
        return ravelIndex(originalIndices, shape: T.ShapeType.dimensionSizes, strides: originalStrides)
    }

    @usableFromInline
    static func unravelIndex(_ index: Int, shape: [Int], strides: [Int]) -> [Int] {
        var result = Array(repeating: 0, count: shape.count)
        var idx = index
        for i in 0..<shape.count {
            result[i] = idx / strides[i]
            idx %= strides[i]
        }
        return result
    }

    @usableFromInline
    static func ravelIndex(_ indices: [Int], shape: [Int], strides: [Int]) -> Int {
        var index = 0
        for i in 0..<indices.count {
            index += indices[i] * strides[i]
        }
        return index
    }
}
