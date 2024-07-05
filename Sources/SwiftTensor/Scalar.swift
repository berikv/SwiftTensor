
public protocol Scalar: Numeric, Comparable, ExpressibleByIntegerLiteral {
    init(_ int: Int)
    static func / (lhs: Self, rhs: Self) -> Self
    static func random(in range: Range<Self>) -> Self
}

extension Float16: Scalar {}
extension Float32: Scalar {}
extension Float64: Scalar {}
extension Int: Scalar {}
extension Int8: Scalar {}
extension Int16: Scalar {}
extension Int32: Scalar {}
extension Int64: Scalar {}
extension UInt: Scalar {}
extension UInt8: Scalar {}
extension UInt16: Scalar {}
extension UInt32: Scalar {}
extension UInt64: Scalar {}
