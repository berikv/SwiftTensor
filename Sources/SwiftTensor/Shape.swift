
public protocol Shape {
    static var dimensionSizes: [Int] { get }
}

extension Shape {
    @inlinable
    public static var dimensionCount: Int { dimensionSizes.count }

    @inlinable
    public static var scalarCount: Int { dimensionSizes.reduce(1, *) }
}
