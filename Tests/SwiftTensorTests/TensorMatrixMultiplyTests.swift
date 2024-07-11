import Testing
import SwiftTensor

struct MatmulTests {
    @Test
    func testMatmulWithIdentityMatrix() {
        let identityMatrix: [Float] = [1, 0, 0, 1]
        let matrix: [Float] = [3, 4, 2, 1]

        let tensor1 = Tensor2x2(identityMatrix)
        let tensor2 = Tensor2x2(matrix)
        var result = Tensor2x2.zero
        Tensor2x2.matrixMultiplying(into: &result, tensor1, tensor2)

        #expect(result.scalars == matrix)
    }

    @Test
    func testMatmulWithCompatibleShapes() {
        let matrixA: [Float] = [1, 2, 3, 4, 5, 6]
        let matrixB: [Float] = [7, 8, 9, 10, 11, 12]
        let expected: [Float] = [58, 64, 139, 154]

        let tensorA = Tensor2x3(matrixA)
        let tensorB = Tensor3x2(matrixB)
        var result = Tensor2x2.zero
        Tensor2x2.matrixMultiplying(into: &result, tensorA, tensorB)

        #expect(result.scalars == expected)
    }

    @Test
    func testMatmulWithLargerMatrices() {
        let matrixA: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        let matrixB: [Float] = [9, 8, 7, 6, 5, 4, 3, 2, 1]
        let expected: [Float] = [30, 24, 18, 84, 69, 54, 138, 114, 90]

        let tensorA = Tensor3x3(matrixA)
        let tensorB = Tensor3x3(matrixB)
        var result = Tensor3x3.zero
        Tensor3x3.matrixMultiplying(into: &result, tensorA, tensorB)

        #expect(result.scalars == expected)
    }

    @Test
    func testMatmulWithTransposeLHS() {
        let matrixA: [Float] = [1, 2, 3, 4, 5, 6]
        let matrixB: [Float] = [7, 8, 9, 10, 11, 12]
        let expected: [Float] = [47, 52, 57, 64, 71, 78, 81, 90, 99]

        let tensorA = Tensor2x3(matrixA)
        let tensorB = Tensor2x3(matrixB)
        var result = Tensor3x3.zero
        Tensor3x3.matrixMultiplying(into: &result, transpose: tensorA, tensorB)

        #expect(result.scalars == expected)
    }

    @Test
    func testMatmulWithTransposeRHS() {
        let matrixA: [Float] = [1, 2, 3, 4, 5, 6]
        let matrixB: [Float] = [7, 8, 9, 10, 11, 12]
        let expected: [Float] = [50, 68, 122, 167]

        let tensorA = Tensor2x3(matrixA)
        let tensorB = Tensor2x3(matrixB)
        var result = Tensor2x2.zero
        Tensor2x2.matrixMultiplying(into: &result, tensorA, transpose: tensorB)

        #expect(result.scalars == expected)
    }
}
