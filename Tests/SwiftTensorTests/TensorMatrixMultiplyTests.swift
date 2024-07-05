import Testing
import SwiftTensor

// Define the shapes for matrices to use in tests
public struct Shape2x2: Shape {
    public static var dimensionSizes: [Int] { [2, 2] }
}

public struct Shape2x3: Shape {
    public static var dimensionSizes: [Int] { [2, 3] }
}

public struct Shape3x2: Shape {
    public static var dimensionSizes: [Int] { [3, 2] }
}

public struct Shape3x3: Shape {
    public static var dimensionSizes: [Int] { [3, 3] }
}

public struct Shape4x4: Shape {
    public static var dimensionSizes: [Int] { [4, 4] }
}

// Define the CPU and SIMD Tensor types to use in tests
typealias CPUTensor2x2 = CPUTensor<Float, Shape2x2>
typealias SIMDTensor2x2 = SIMDTensor<Float, Shape2x2>
typealias CPUTensor2x3 = CPUTensor<Float, Shape2x3>
typealias SIMDTensor2x3 = SIMDTensor<Float, Shape2x3>
typealias CPUTensor3x2 = CPUTensor<Float, Shape3x2>
typealias SIMDTensor3x2 = SIMDTensor<Float, Shape3x2>
typealias CPUTensor3x3 = CPUTensor<Float, Shape3x3>
typealias SIMDTensor3x3 = SIMDTensor<Float, Shape3x3>
typealias CPUTensor4x4 = CPUTensor<Float, Shape4x4>
typealias SIMDTensor4x4 = SIMDTensor<Float, Shape4x4>

struct MatmulTests {
    @Test
    func testMatmulWithIdentityMatrix() {
        let identityMatrix: [Float] = [1, 0, 0, 1] // 2x2 identity matrix
        let matrix: [Float] = [3, 4, 2, 1] // 2x2 matrix

        let tensor1 = CPUTensor2x2(identityMatrix)
        let tensor2 = CPUTensor2x2(matrix)
        let result = CPUTensor2x2.matrixMultiply(lhs: tensor1, rhs: tensor2)

        #expect(result.scalars == matrix)
    }

    @Test
    func testMatmulWithZeroMatrix() {
        let zeroMatrix: [Float] = [0, 0, 0, 0] // 2x2 zero matrix
        let matrix: [Float] = [3, 4, 2, 1] // 2x2 matrix
        let expected: [Float] = [0, 0, 0, 0]

        let tensor1 = SIMDTensor2x2(zeroMatrix)
        let tensor2 = SIMDTensor2x2(matrix)
        let result = SIMDTensor2x2.matrixMultiply(lhs: tensor1, rhs: tensor2)

        #expect(result.scalars == expected)
    }

    @Test
    func testMatmulWithCompatibleShapes() {
        let matrixA: [Float] = [1, 2, 3, 4, 5, 6] // 2x3 matrix
        let matrixB: [Float] = [7, 8, 9, 10, 11, 12] // 3x2 matrix
        let expected: [Float] = [58, 64, 139, 154] // 2x2 result

        let tensorA = CPUTensor2x3(matrixA)
        let tensorB = CPUTensor3x2(matrixB)
        let result = CPUTensor2x2.matrixMultiply(lhs: tensorA, rhs: tensorB)

        #expect(result.scalars == expected)
    }

    @Test
    func testMatmulWithNonSquareMatrices() {
        let matrixA: [Float] = [1, 2, 3, 4, 5, 6] // 2x3 matrix
        let matrixB: [Float] = [7, 8, 9, 10, 11, 12] // 3x2 matrix
        let expected: [Float] = [58, 64, 139, 154] // 2x2 result

        let tensorA = SIMDTensor2x3(matrixA)
        let tensorB = SIMDTensor3x2(matrixB)
        let result = SIMDTensor2x2.matrixMultiply(lhs: tensorA, rhs: tensorB)

        #expect(result.scalars == expected)
    }

    @Test
    func testMatmulWithLargerMatrices() {
        let matrixA: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9] // 3x3 matrix
        let matrixB: [Float] = [9, 8, 7, 6, 5, 4, 3, 2, 1] // 3x3 matrix
        let expected: [Float] = [30, 24, 18, 84, 69, 54, 138, 114, 90] // 3x3 result

        let tensorA = CPUTensor3x3(matrixA)
        let tensorB = CPUTensor3x3(matrixB)
        let result = CPUTensor3x3.matrixMultiply(lhs: tensorA, rhs: tensorB)

        #expect(result.scalars == expected)
    }

    @Test
    func testMatmulWith4x4Matrices() {
        let matrixA: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] // 4x4 matrix
        let matrixB: [Float] = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1] // 4x4 matrix
        let expected: [Float] = [80, 70, 60, 50, 240, 214, 188, 162, 400, 358, 316, 274, 560, 502, 444, 386] // 4x4 result

        let tensorA = SIMDTensor4x4(matrixA)
        let tensorB = SIMDTensor4x4(matrixB)
        let result = SIMDTensor4x4.matrixMultiply(lhs: tensorA, rhs: tensorB)

        #expect(result.scalars == expected)
    }
}
