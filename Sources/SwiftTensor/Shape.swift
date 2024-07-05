
public protocol Shape {
    static var dimensionSizes: [Int] { get }
}

extension Shape {
    @inlinable
    public static var dimensionCount: Int { dimensionSizes.count }

    @inlinable
    public static var scalarCount: Int { dimensionSizes.reduce(1, *) }
}

@frozen
public struct Shape784: Shape {
    @inlinable
    public static var dimensionSizes: [Int] { [784] }
}

@frozen
public struct Shape28x28: Shape {
    @inlinable
    public static var dimensionSizes: [Int] { [28, 28] }
}
