
public protocol Tensor {
    associatedtype ShapeType: Shape
    associatedtype ScalarType: Scalar

    var scalars: [ScalarType] { get }

    // MARK: Initialisation

    init(_ scalars: [ScalarType])
    init(repeating scalar: ScalarType)
    static func random(in range: Range<ScalarType>) -> Self

    subscript(index: Int) -> ScalarType { get set }
}
