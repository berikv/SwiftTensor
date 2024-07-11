import Testing
import SwiftTensor

struct TensorMicroBenchmarkTests {

    typealias ShapeType = Shape784
    typealias TensorType = Tensor<ShapeType>

    init() {
#if !TEST_PERFORMANCE
        fatalError("Run performance tests in release builds only")
#endif
    }

    @Test
    func testAdd() {
        let scalars = (0..<ShapeType.scalarCount).map { Float($0) }
        let term = TensorType(scalars)

        let measurement = Measure(name: "Add") {
            var tensor = TensorType(scalars)
            for _ in 0..<1_000 {
                tensor.add(term)
            }
        }

        print(measurement)
        #expect(measurement.mean.toBeCloseTo(0.00047, margin: 0.00005))
    }

    @Test
    func testAdding() {
        let scalars = (0..<ShapeType.scalarCount).map { Float($0) }
        let termA = TensorType(scalars)
        let termB = TensorType(scalars.reversed())
        var result = TensorType.zero

        let measurement = Measure(name: "Adding") {
            for _ in 0..<1_000 {
                TensorType.adding(into: &result, termA, termB)
            }
        }

        print(measurement)
        #expect(measurement.mean.toBeCloseTo(0.00072, margin: 0.0001))
    }

    @Test
    func testSubtract() {
        let scalars = (0..<ShapeType.scalarCount).map { Float($0) }
        let term = TensorType(scalars)

        let measurement = Measure(name: "Subtract") {
            var tensor = TensorType(scalars)
            for _ in 0..<1_000 {
                tensor.subtract(term)
            }
        }

        print(measurement)
        #expect(measurement.mean.toBeCloseTo(0.00050, margin: 0.00005))
    }

    @Test
    func testSubtracting() {
        let scalars = (0..<ShapeType.scalarCount).map { Float($0) }
        let termA = TensorType(scalars)
        let termB = TensorType(scalars.reversed())
        var result = TensorType.zero

        let measurement = Measure(name: "Subtracting") {
            for _ in 0..<1_000 {
                TensorType.subtracting(into: &result, termA, termB)
            }
        }

        print(measurement)
        #expect(measurement.mean.toBeCloseTo(0.00072, margin: 0.0001))
    }

    @Test
    func testMultiply() {
        let scalars = (0..<ShapeType.scalarCount).map { Float($0) }
        let factor = TensorType(scalars)

        let measurement = Measure(name: "Multiply") {
            var tensor = TensorType(scalars)
            for _ in 0..<1_000 {
                tensor.multiply(by: factor)
            }
        }

        print(measurement)
        #expect(measurement.mean.toBeCloseTo(0.00047, margin: 0.00005))
    }

    @Test
    func testMultiplying() {
        let scalars = (0..<ShapeType.scalarCount).map { Float($0) }
        let termA = TensorType(scalars)
        let termB = TensorType(scalars.reversed())
        var result = TensorType.zero

        let measurement = Measure(name: "Multiplying") {
            for _ in 0..<1_000 {
                TensorType.multiplying(into: &result, termA, termB)
            }
        }

        print(measurement)
        #expect(measurement.mean.toBeCloseTo(0.00072, margin: 0.0001))
    }

    @Test
    func testDivide() {
        let scalars = (0..<ShapeType.scalarCount).map { Float(1 + $0) }
        let divisor = TensorType(scalars.reversed())

        let measurement = Measure(name: "Divide") {
            var tensor = TensorType(scalars)
            for _ in 0..<1_000 {
                tensor.divide(by: divisor)
            }
        }

        print(measurement)
        #expect(measurement.mean.toBeCloseTo(0.00047, margin: 0.0001))
    }

    @Test
    func testDividing() {
        let scalars = (0..<ShapeType.scalarCount).map { Float(1 + $0) }
        let termA = TensorType(scalars)
        let termB = TensorType(scalars.reversed())
        var result = TensorType.zero

        let measurement = Measure(name: "Dividing") {
            for _ in 0..<1_000 {
                TensorType.dividing(into: &result, termA, termB)
            }
        }

        print(measurement)
        #expect(measurement.mean.toBeCloseTo(0.00072, margin: 0.0001))
    }

    @Test
    func testSum() {
        let scalars = (0..<ShapeType.scalarCount).map { Float($0) }
        let tensor = TensorType(scalars)
        var result = Float.zero

        let measurement = Measure(name: "Sum") {
            for _ in 0..<10_000 {
                result = tensor.sum()
            }
        }

        print(measurement)
        #expect(result == 306936)
        #expect(measurement.mean.toBeCloseTo(0.0004, margin: 0.00005))
    }

    @Test
    func testMean() {
        let scalars = (0..<ShapeType.scalarCount).map { Float($0) }
        let tensor = TensorType(scalars)
        var result = Float.zero

        let measurement = Measure(name: "Mean") {
            for _ in 0..<100_000 {
                result += tensor.mean()
            }
        }

        print(measurement)
        #expect(result.toBeCloseTo(5.4032864e+08))
        #expect(measurement.mean.toBeCloseTo(0.0043, margin: 0.0005))
    }

    @Test
    func testMax() {
        let scalars = (0..<ShapeType.scalarCount).map { Float($0) }
        let tensor = TensorType(scalars)
        var result = Float.zero

        let measurement = Measure(name: "Max") {
            for _ in 0..<1_000 {
                result = tensor.max()
            }
        }

        print(measurement)
        #expect(result == Float(ShapeType.scalarCount - 1))
        #expect(measurement.mean.toBeCloseTo(0.001, margin: 0.0005))
    }

    @Test
    func testExp() {
        let scalars = (0..<ShapeType.scalarCount).map { Float($0) }
        var result = TensorType.zero

        let measurement = Measure(name: "Exp") {
            for _ in 0..<1_000 {
                var tensor = TensorType(scalars)
                tensor.exp()
                result = tensor
            }
        }

        print(measurement)
        #expect(result.scalars[4].toBeCloseTo(54.598, margin: 0.001))
        #expect(measurement.mean.toBeCloseTo(0.0017, margin: 0.0005))
    }

    @Test
    func testReLu() {
        let scalars = (0..<ShapeType.scalarCount).map { $0.isMultiple(of: 2) ? Float($0) : -Float($0) }
        var result = TensorType.zero

        let measurement = Measure(name: "ReLu") {
            for _ in 0..<1_000 {
                var tensor = TensorType(scalars)
                tensor.relu()
                result = tensor
            }
        }

        print(measurement)
        #expect(result.scalars[4] == 4)
        #expect(result.scalars[5] == 0)
        #expect(measurement.mean.toBeCloseTo(0.00098, margin: 0.0001))
    }

    @Test
    func testSoftmax() {
        let scalars = (0..<ShapeType.scalarCount).map { $0.isMultiple(of: 2) ? Float($0) : -Float($0) }
        var result = TensorType.zero

        let measurement = Measure(name: "Softmax") {
            for _ in 0..<1_000 {
                var tensor = TensorType(scalars)
                tensor.softmax()
                result = tensor
            }
        }

        print(measurement)
        #expect(abs(result.sum() - 1) < 1e-6)
        #expect(measurement.mean.toBeCloseTo(0.0048, margin: 0.0005))
    }

    @Test
    func testClip() {
        let scalars = (0..<ShapeType.scalarCount).map { Float($0) }
        var result = TensorType.zero

        let measurement = Measure(name: "Clip") {
            for _ in 0..<1_000 {
                var tensor = TensorType(scalars)
                tensor.clip(to: 1)
                result = tensor
            }
        }

        print(measurement)
        #expect(result.scalars[10] == 1)
        #expect(measurement.mean.toBeCloseTo(0.00047, margin: 0.00008))
    }

    @Test
    func testApplyL2() {
        let scalars = (0..<ShapeType.scalarCount).map { Float($0) }
        var result = TensorType.zero

        let measurement = Measure(name: "ApplyL2") {
            for _ in 0..<1_000 {
                var tensor = TensorType(scalars)
                tensor.applyL2(lambda: 0.0001)
                result = tensor
            }
        }

        print(measurement)
        #expect(result.scalars[10].toBeCloseTo(9.999))
        #expect(measurement.mean.toBeCloseTo(0.00037, margin: 0.0001))
    }

    @Test
    func testMatrixMultiplying() {
        let matrixA: [Float] = [1, 2, 3, 4, 5, 6]
        let matrixB: [Float] = [7, 8, 9, 10, 11, 12]
        let expected: [Float] = [58, 64, 139, 154]

        let tensorA = Tensor2x3(matrixA)
        let tensorB = Tensor3x2(matrixB)
        var result = Tensor2x2.zero

        let measurement = Measure(name: "Matmul") {
            for _ in 0..<100_000 {
                Tensor2x2.matrixMultiplying(into: &result, tensorA, tensorB)
            }
        }

        print(measurement)
        #expect(result.scalars == expected)
        #expect(measurement.mean.toBeCloseTo(0.0012, margin: 0.0005))
    }

    @Test
    func testMatrixMultiplyingWithTransposeLHS() {
        let matrixA: [Float] = [1, 2, 3, 4, 5, 6]
        let matrixB: [Float] = [7, 8, 9, 10, 11, 12]
        let expected: [Float] = [47, 52, 57, 64, 71, 78, 81, 90, 99]

        let tensorA = Tensor2x3(matrixA)
        let tensorB = Tensor2x3(matrixB)
        var result = Tensor3x3.zero

        let measurement = Measure(name: "Matmul") {
            for _ in 0..<100_000 {
                Tensor3x3.matrixMultiplying(into: &result, transpose: tensorA, tensorB)
            }
        }

        print(measurement)
        #expect(result.scalars == expected)
        #expect(measurement.mean.toBeCloseTo(0.0022, margin: 0.001))
    }

    @Test
    func testMatrixMultiplyingWithTransposeRHS() {
        let matrixA: [Float] = [1, 2, 3, 4, 5, 6]
        let matrixB: [Float] = [7, 8, 9, 10, 11, 12]
        let expected: [Float] = [50, 68, 122, 167]

        let tensorA = Tensor2x3(matrixA)
        let tensorB = Tensor2x3(matrixB)
        var result = Tensor2x2.zero

        let measurement = Measure(name: "Matmul") {
            for _ in 0..<100_000 {
                Tensor2x2.matrixMultiplying(into: &result, tensorA, transpose: tensorB)
            }
        }

        print(measurement)
        #expect(result.scalars == expected)
        #expect(measurement.mean.toBeCloseTo(0.0012, margin: 0.0005))
    }
}
