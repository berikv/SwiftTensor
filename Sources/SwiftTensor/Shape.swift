
public protocol Shape {
    static var dimensionSizes: [Int] { get }
    static var strides: [Int] { get }
}

extension Shape {
    @inlinable
    public static var dimensionCount: Int { dimensionSizes.count }

    @inlinable
    public static var scalarCount: Int { dimensionSizes.reduce(1, *) }
}

public struct ShapeDual<S: Shape>: Shape {
    @inlinable
    public static var dimensionSizes: [Int] { S.dimensionSizes.reversed() }
    @inlinable
    public static var strides: [Int] { computeStrides() }
}

@frozen
public struct Shape784: Shape {
    @inlinable
    public static var dimensionSizes: [Int] { [784] }
    public static let strides: [Int] = computeStrides()
}

@frozen
public struct Shape28x28: Shape {
    @inlinable
    public static var dimensionSizes: [Int] { [28, 28] }
    public static let strides: [Int] = computeStrides()
}
