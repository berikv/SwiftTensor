import simd

@inlinable
public func simd_min<S: SIMD>(_ first: S, _ second: S) -> S {
    switch S.self {
    case is SIMD2<Float16>.Type:
        return simd.simd_min(first as! SIMD2<Float16>, second as! SIMD2<Float16>) as! S
    case is SIMD4<Float16>.Type:
        return simd.simd_min(first as! SIMD4<Float16>, second as! SIMD4<Float16>) as! S
    case is SIMD8<Float16>.Type:
        return simd.simd_min(first as! SIMD8<Float16>, second as! SIMD8<Float16>) as! S
    case is SIMD16<Float16>.Type:
        return simd.simd_min(first as! SIMD16<Float16>, second as! SIMD16<Float16>) as! S
    case is SIMD32<Float16>.Type:
        return simd.simd_min(first as! SIMD32<Float16>, second as! SIMD32<Float16>) as! S
    case is SIMD2<Float>.Type:
        return simd.simd_min(first as! SIMD2<Float>, second as! SIMD2<Float>) as! S
    case is SIMD4<Float>.Type:
        return simd.simd_min(first as! SIMD4<Float>, second as! SIMD4<Float>) as! S
    case is SIMD8<Float>.Type:
        return simd.simd_min(first as! SIMD8<Float>, second as! SIMD8<Float>) as! S
    case is SIMD16<Float>.Type:
        return simd.simd_min(first as! SIMD16<Float>, second as! SIMD16<Float>) as! S
    case is SIMD2<Double>.Type:
        return simd.simd_min(first as! SIMD2<Double>, second as! SIMD2<Double>) as! S
    case is SIMD4<Double>.Type:
        return simd.simd_min(first as! SIMD4<Double>, second as! SIMD4<Double>) as! S
    case is SIMD8<Double>.Type:
        return simd.simd_min(first as! SIMD8<Double>, second as! SIMD8<Double>) as! S
    case is SIMD2<Int>.Type:
        return simd.simd_min(first as! SIMD2<Int>, second as! SIMD2<Int>) as! S
    case is SIMD4<Int>.Type:
        return simd.simd_min(first as! SIMD4<Int>, second as! SIMD4<Int>) as! S
    case is SIMD8<Int>.Type:
        return simd.simd_min(first as! SIMD8<Int>, second as! SIMD8<Int>) as! S
    case is SIMD2<UInt>.Type:
        return simd.simd_min(first as! SIMD2<UInt>, second as! SIMD2<UInt>) as! S
    case is SIMD4<UInt>.Type:
        return simd.simd_min(first as! SIMD4<UInt>, second as! SIMD4<UInt>) as! S
    case is SIMD8<UInt>.Type:
        return simd.simd_min(first as! SIMD8<UInt>, second as! SIMD8<UInt>) as! S
    default:
        fatalError("Unsupported SIMD type")
    }
}

@inlinable
public func simd_max<S: SIMD>(_ first: S, _ second: S) -> S {
    switch S.self {
    case is SIMD2<Float16>.Type:
        return simd.simd_max(first as! SIMD2<Float16>, second as! SIMD2<Float16>) as! S
    case is SIMD4<Float16>.Type:
        return simd.simd_max(first as! SIMD4<Float16>, second as! SIMD4<Float16>) as! S
    case is SIMD8<Float16>.Type:
        return simd.simd_max(first as! SIMD8<Float16>, second as! SIMD8<Float16>) as! S
    case is SIMD16<Float16>.Type:
        return simd.simd_max(first as! SIMD16<Float16>, second as! SIMD16<Float16>) as! S
    case is SIMD32<Float16>.Type:
        return simd.simd_max(first as! SIMD32<Float16>, second as! SIMD32<Float16>) as! S
    case is SIMD2<Float>.Type:
        return simd.simd_max(first as! SIMD2<Float>, second as! SIMD2<Float>) as! S
    case is SIMD4<Float>.Type:
        return simd.simd_max(first as! SIMD4<Float>, second as! SIMD4<Float>) as! S
    case is SIMD8<Float>.Type:
        return simd.simd_max(first as! SIMD8<Float>, second as! SIMD8<Float>) as! S
    case is SIMD16<Float>.Type:
        return simd.simd_max(first as! SIMD16<Float>, second as! SIMD16<Float>) as! S
    case is SIMD2<Double>.Type:
        return simd.simd_max(first as! SIMD2<Double>, second as! SIMD2<Double>) as! S
    case is SIMD4<Double>.Type:
        return simd.simd_max(first as! SIMD4<Double>, second as! SIMD4<Double>) as! S
    case is SIMD8<Double>.Type:
        return simd.simd_max(first as! SIMD8<Double>, second as! SIMD8<Double>) as! S
    case is SIMD2<Int>.Type:
        return simd.simd_max(first as! SIMD2<Int>, second as! SIMD2<Int>) as! S
    case is SIMD4<Int>.Type:
        return simd.simd_max(first as! SIMD4<Int>, second as! SIMD4<Int>) as! S
    case is SIMD8<Int>.Type:
        return simd.simd_max(first as! SIMD8<Int>, second as! SIMD8<Int>) as! S
    case is SIMD2<UInt>.Type:
        return simd.simd_max(first as! SIMD2<UInt>, second as! SIMD2<UInt>) as! S
    case is SIMD4<UInt>.Type:
        return simd.simd_max(first as! SIMD4<UInt>, second as! SIMD4<UInt>) as! S
    case is SIMD8<UInt>.Type:
        return simd.simd_max(first as! SIMD8<UInt>, second as! SIMD8<UInt>) as! S
    default:
        fatalError("Unsupported SIMD type")
    }
}

@inlinable
public func simd_reduce_max<S: SIMD>(_ values: S) -> S.Scalar {
    switch S.self {
    case is SIMD2<Float16>.Type:
        return simd.simd_reduce_max(values as! SIMD2<Float16>) as! S.Scalar
    case is SIMD4<Float16>.Type:
        return simd.simd_reduce_max(values as! SIMD4<Float16>) as! S.Scalar
    case is SIMD8<Float16>.Type:
        return simd.simd_reduce_max(values as! SIMD8<Float16>) as! S.Scalar
    case is SIMD16<Float16>.Type:
        return simd.simd_reduce_max(values as! SIMD16<Float16>) as! S.Scalar
    case is SIMD32<Float16>.Type:
        return simd.simd_reduce_max(values as! SIMD32<Float16>) as! S.Scalar
    case is SIMD2<Float>.Type:
        return simd.simd_reduce_max(values as! SIMD2<Float>) as! S.Scalar
    case is SIMD4<Float>.Type:
        return simd.simd_reduce_max(values as! SIMD4<Float>) as! S.Scalar
    case is SIMD8<Float>.Type:
        return simd.simd_reduce_max(values as! SIMD8<Float>) as! S.Scalar
    case is SIMD16<Float>.Type:
        return simd.simd_reduce_max(values as! SIMD16<Float>) as! S.Scalar
    case is SIMD2<Double>.Type:
        return simd.simd_reduce_max(values as! SIMD2<Double>) as! S.Scalar
    case is SIMD4<Double>.Type:
        return simd.simd_reduce_max(values as! SIMD4<Double>) as! S.Scalar
    case is SIMD8<Double>.Type:
        return simd.simd_reduce_max(values as! SIMD8<Double>) as! S.Scalar
    case is SIMD2<Int>.Type:
        return simd.simd_reduce_max(values as! SIMD2<Int>) as! S.Scalar
    case is SIMD4<Int>.Type:
        return simd.simd_reduce_max(values as! SIMD4<Int>) as! S.Scalar
    case is SIMD8<Int>.Type:
        return simd.simd_reduce_max(values as! SIMD8<Int>) as! S.Scalar
    case is SIMD2<UInt>.Type:
        return simd.simd_reduce_max(values as! SIMD2<UInt>) as! S.Scalar
    case is SIMD4<UInt>.Type:
        return simd.simd_reduce_max(values as! SIMD4<UInt>) as! S.Scalar
    case is SIMD8<UInt>.Type:
        return simd.simd_reduce_max(values as! SIMD8<UInt>) as! S.Scalar
    default:
        fatalError("Unsupported SIMD type")
    }
}



@inlinable
public func exp<S: SIMD>(_ x: S) -> S {
    switch S.self {
    case is SIMD2<Float16>.Type:
        return SIMD2<Float16>(x.indices.map { Float16(exp(Float(x[$0] as! Float16))) }) as! S
    case is SIMD4<Float16>.Type:
        let converted = SIMD4<Float>(x.indices.map { Float(x[$0] as! Float16) })
        return simd.exp(converted).indices.map { Float16($0) } as! S
    case is SIMD8<Float16>.Type:
        let converted = SIMD8<Float>(x.indices.map { Float(x[$0] as! Float16) })
        return simd.exp(converted).indices.map { Float16($0) } as! S
    case is SIMD16<Float16>.Type:
        let converted = SIMD16<Float>(x.indices.map { Float(x[$0] as! Float16) })
        return simd.exp(converted).indices.map { Float16($0) } as! S
    case is SIMD2<Float>.Type:
        return simd.exp(x as! SIMD2<Float>) as! S
    case is SIMD4<Float>.Type:
        return simd.exp(x as! SIMD4<Float>) as! S
    case is SIMD8<Float>.Type:
        return simd.exp(x as! SIMD8<Float>) as! S
    case is SIMD16<Float>.Type:
        return simd.exp(x as! SIMD16<Float>) as! S
    case is SIMD2<Double>.Type:
        return simd.exp(x as! SIMD2<Double>) as! S
    case is SIMD4<Double>.Type:
        return simd.exp(x as! SIMD4<Double>) as! S
    case is SIMD8<Double>.Type:
        return simd.exp(x as! SIMD8<Double>) as! S
    default:
        fatalError("Unsupported SIMD type")
    }
}

public protocol SIMDStepCompatible: SIMDScalar, Codable, Hashable {}
extension Float16: SIMDStepCompatible {}
extension Float: SIMDStepCompatible {}
extension Double: SIMDStepCompatible {}

@inlinable
public func simd_step<S: SIMD>(edge: S, x: S) -> S where S.Scalar: SIMDStepCompatible {
    switch (S.Scalar.self, S.scalarCount) {
    case (is Float16.Type, 2):
        return simd.simd_step(edge as! SIMD2<Float16>, x as! SIMD2<Float16>) as! S
    case (is Float16.Type, 3):
        return simd.simd_step(edge as! SIMD3<Float16>, x as! SIMD3<Float16>) as! S
    case (is Float16.Type, 4):
        return simd.simd_step(edge as! SIMD4<Float16>, x as! SIMD4<Float16>) as! S
    case (is Float16.Type, 8):
        return simd.simd_step(edge as! SIMD8<Float16>, x as! SIMD8<Float16>) as! S
    case (is Float16.Type, 16):
        return simd.simd_step(edge as! SIMD16<Float16>, x as! SIMD16<Float16>) as! S
    case (is Float16.Type, 32):
        return simd.simd_step(edge as! SIMD32<Float16>, x as! SIMD32<Float16>) as! S

    case (is Float.Type, 2):
        return simd.simd_step(edge as! SIMD2<Float>, x as! SIMD2<Float>) as! S
    case (is Float.Type, 3):
        return simd.simd_step(edge as! SIMD3<Float>, x as! SIMD3<Float>) as! S
    case (is Float.Type, 4):
        return simd.simd_step(edge as! SIMD4<Float>, x as! SIMD4<Float>) as! S
    case (is Float.Type, 8):
        return simd.simd_step(edge as! SIMD8<Float>, x as! SIMD8<Float>) as! S
    case (is Float.Type, 16):
        return simd.simd_step(edge as! SIMD16<Float>, x as! SIMD16<Float>) as! S

    case (is Double.Type, 2):
        return simd.simd_step(edge as! SIMD2<Double>, x as! SIMD2<Double>) as! S
    case (is Double.Type, 3):
        return simd.simd_step(edge as! SIMD3<Double>, x as! SIMD3<Double>) as! S
    case (is Double.Type, 4):
        return simd.simd_step(edge as! SIMD4<Double>, x as! SIMD4<Double>) as! S
    case (is Double.Type, 8):
        return simd.simd_step(edge as! SIMD8<Double>, x as! SIMD8<Double>) as! S

    default:
        fatalError("Unsupported SIMD type")
    }
}
