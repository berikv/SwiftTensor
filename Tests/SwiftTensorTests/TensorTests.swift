import Testing
import SwiftTensor
import Darwin

struct Tensor_Shape784_Tests {
    typealias ShapeType = Shape784
    typealias ScalarType = Float
    typealias TensorType = Tensor<ShapeType>

    @Test
    func testInitWithScalars() {
        let scalars = (0..<ShapeType.scalarCount).map { ScalarType($0) }
        let tensor = TensorType(scalars)
        #expect(tensor.scalars.count == ShapeType.scalarCount)
    }

    @Test
    func testInitWithRepeatingValue() {
        let value: ScalarType = ScalarType(3.14)
        let tensor = TensorType(repeating: value)
        #expect(tensor.scalars == [ScalarType](repeating: value, count: ShapeType.scalarCount))
    }

    @Test
    func testInitWithRandomRange() {
        let tensor = TensorType.random(in: 0..<1)
        let inRange = tensor.scalars.allSatisfy { $0 >= 0 && $0 <= 1 }
        #expect(inRange)
    }

    @Test
    func testSubscriptGet() {
        let tensor = TensorType([2, 3, 4] + Array<ScalarType>(repeating: 0, count: ShapeType.scalarCount - 3))
        #expect(tensor[0] == 2)
        #expect(tensor[1] == 3)
        #expect(tensor[2] == 4)
    }

    @Test
    func testSubscriptSet() {
        var tensor = TensorType([2, 3, 4] + Array<ScalarType>(repeating: 0, count: ShapeType.scalarCount - 3))
        tensor[3] = 8
        #expect(tensor[3] == 8)
    }

    @Test
    func testAdding() {
        let tensor1 = TensorType(repeating: 1)
        let tensor2 = TensorType(repeating: 2)
        var result = TensorType.zero
        TensorType.adding(into: &result, tensor1, tensor2)
        #expect(result.scalars == [ScalarType](repeating: 3, count: ShapeType.scalarCount))
    }

    @Test
    func testSubtracting() {
        let tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        var result = TensorType.zero
        TensorType.subtracting(into: &result, tensor1, tensor2)
        #expect(result.scalars == [ScalarType](repeating: 1, count: ShapeType.scalarCount))
    }

    @Test
    func testMultiplying() {
        let tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        var result = TensorType.zero
        TensorType.multiplying(into: &result, tensor1, tensor2)
        #expect(result.scalars == [ScalarType](repeating: 6, count: ShapeType.scalarCount))
    }

    @Test
    func testDividing() {
        let tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        var result = TensorType.zero
        TensorType.dividing(into: &result, tensor1, tensor2)
        #expect(result.scalars == [ScalarType](repeating: 1.5, count: ShapeType.scalarCount))
    }

    @Test
    func testAdd() {
        var tensor1 = TensorType(repeating: 1)
        let tensor2 = TensorType(repeating: 2)
        tensor1.add(tensor2)
        #expect(tensor1.scalars == [ScalarType](repeating: 3, count: ShapeType.scalarCount))
    }

    @Test
    func testSubtract() {
        var tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        tensor1.subtract(tensor2)
        #expect(tensor1.scalars == [ScalarType](repeating: 1, count: ShapeType.scalarCount))
    }

    @Test
    func testMultiply() {
        var tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        tensor1.multiply(by: tensor2)
        #expect(tensor1.scalars == [ScalarType](repeating: 6, count: ShapeType.scalarCount))
    }

    @Test
    func testDivide() {
        var tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        tensor1.divide(by: tensor2)
        #expect(tensor1.scalars == [ScalarType](repeating: 1.5, count: ShapeType.scalarCount))
    }

    //    @Test
    //    func testMin() {
    //        var tensor = TensorType(repeating: 10)
    //        let someIndex = 5
    //        tensor[someIndex] = 5
    //        #expect(tensor.min() == 5)
    //    }

    @Test
    func testMax() {
        var tensor = TensorType(repeating: -10)
        let someIndex = 3
        tensor[someIndex] = -5
        #expect(tensor.max() == -5)
    }

    @Test
    func testSum() {
        let scalars: [ScalarType] = [1, 2, 3, 4, 5]
        let tensor = TensorType(scalars + [ScalarType](repeating: 0, count: ShapeType.scalarCount - scalars.count))
        let result = tensor.sum()
        #expect(result == scalars.reduce(0, +))
    }

    @Test
    func testMean() {
        let scalarCount = ShapeType.scalarCount
        let scalars = [ScalarType](repeating: 5, count: scalarCount / 2)
        + [ScalarType](repeating: 7, count: scalarCount / 2)
        + (scalarCount.isMultiple(of: 2) ? [] : [6])

        let tensor = TensorType(scalars)

        #expect(tensor.mean() == 6)
    }

    @Test
    func testExp() {
        let scalars: [ScalarType] = [0, 1, 2, 3, 4]
        var tensor = TensorType(scalars + [ScalarType](repeating: 0, count: ShapeType.scalarCount - scalars.count))

        let expectedScalars = scalars.map { ScalarType(Darwin.exp($0)) }
        + [ScalarType](repeating: ScalarType(Darwin.exp(Double(0))), count: ShapeType.scalarCount - scalars.count)

        tensor.exp()
        #expect(tensor.scalars == expectedScalars)
    }

    @Test
    func testSoftmax() {
        let scalars: [ScalarType] = [0, 1, -2, 3, 4]
        var tensor = TensorType(scalars + [ScalarType](repeating: 1, count: ShapeType.scalarCount - scalars.count))

        tensor.softmax()

        for scalar in tensor.scalars {
            #expect(scalar <= 1)
            #expect(scalar >= 0)
        }

        #expect(abs(tensor.sum() - 1) < 1e-6)
    }

    @Test
    func testRelu() {
        let scalars: [ScalarType] = [-1, -0.5, 0, 0.5, 1]
        let padding = [ScalarType](repeating: 0, count: ShapeType.scalarCount - scalars.count)
        var tensor = TensorType(scalars + padding)
        tensor.relu()

        let expectedScalars: [ScalarType] = [0, 0, 0, 0.5, 1] + padding
        #expect(tensor.scalars == expectedScalars)
    }
}

struct Tensor_Shape15_Tests {
    typealias ShapeType = Shape15
    typealias ScalarType = Float
    typealias TensorType = Tensor<ShapeType>

    @Test
    func testInitWithScalars() {
        let scalars = (0..<ShapeType.scalarCount).map { ScalarType($0) }
        let tensor = TensorType(scalars)
        #expect(tensor.scalars.count == ShapeType.scalarCount)
    }

    @Test
    func testInitWithRepeatingValue() {
        let value: ScalarType = ScalarType(3.14)
        let tensor = TensorType(repeating: value)
        #expect(tensor.scalars == [ScalarType](repeating: value, count: ShapeType.scalarCount))
    }

    @Test
    func testInitWithRandomRange() {
        let tensor = TensorType.random(in: 0..<1)
        let inRange = tensor.scalars.allSatisfy { $0 >= 0 && $0 <= 1 }
        #expect(inRange)
    }

    @Test
    func testSubscriptGet() {
        let tensor = TensorType([2, 3, 4] + Array<ScalarType>(repeating: 0, count: ShapeType.scalarCount - 3))
        #expect(tensor[0] == 2)
        #expect(tensor[1] == 3)
        #expect(tensor[2] == 4)
    }

    @Test
    func testSubscriptSet() {
        var tensor = TensorType([2, 3, 4] + Array<ScalarType>(repeating: 0, count: ShapeType.scalarCount - 3))
        tensor[3] = 8
        #expect(tensor[3] == 8)
    }

    @Test
    func testAdding() {
        let tensor1 = TensorType(repeating: 1)
        let tensor2 = TensorType(repeating: 2)
        var result = TensorType.zero
        TensorType.adding(into: &result, tensor1, tensor2)
        #expect(result.scalars == [ScalarType](repeating: 3, count: ShapeType.scalarCount))
    }

    @Test
    func testSubtracting() {
        let tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        var result = TensorType.zero
        TensorType.subtracting(into: &result, tensor1, tensor2)
        #expect(result.scalars == [ScalarType](repeating: 1, count: ShapeType.scalarCount))
    }

    @Test
    func testMultiplying() {
        let tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        var result = TensorType.zero
        TensorType.multiplying(into: &result, tensor1, tensor2)
        #expect(result.scalars == [ScalarType](repeating: 6, count: ShapeType.scalarCount))
    }

    @Test
    func testDividing() {
        let tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        var result = TensorType.zero
        TensorType.dividing(into: &result, tensor1, tensor2)
        #expect(result.scalars == [ScalarType](repeating: 1.5, count: ShapeType.scalarCount))
    }

    @Test
    func testAdd() {
        var tensor1 = TensorType(repeating: 1)
        let tensor2 = TensorType(repeating: 2)
        tensor1.add(tensor2)
        #expect(tensor1.scalars == [ScalarType](repeating: 3, count: ShapeType.scalarCount))
    }

    @Test
    func testSubtract() {
        var tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        tensor1.subtract(tensor2)
        #expect(tensor1.scalars == [ScalarType](repeating: 1, count: ShapeType.scalarCount))
    }

    @Test
    func testMultiply() {
        var tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        tensor1.multiply(by: tensor2)
        #expect(tensor1.scalars == [ScalarType](repeating: 6, count: ShapeType.scalarCount))
    }

    @Test
    func testDivide() {
        var tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        tensor1.divide(by: tensor2)
        #expect(tensor1.scalars == [ScalarType](repeating: 1.5, count: ShapeType.scalarCount))
    }

    //    @Test
    //    func testMin() {
    //        var tensor = TensorType(repeating: 10)
    //        let someIndex = 5
    //        tensor[someIndex] = 5
    //        #expect(tensor.min() == 5)
    //    }

    @Test
    func testMax() {
        var tensor = TensorType(repeating: -10)
        let someIndex = 3
        tensor[someIndex] = -5
        #expect(tensor.max() == -5)
    }

    @Test
    func testSum() {
        let scalars: [ScalarType] = [1, 2, 3, 4, 5]
        let tensor = TensorType(scalars + [ScalarType](repeating: 0, count: ShapeType.scalarCount - scalars.count))
        let result = tensor.sum()
        #expect(result == scalars.reduce(0, +))
    }

    @Test
    func testMean() {
        let scalarCount = ShapeType.scalarCount
        let scalars = [ScalarType](repeating: 5, count: scalarCount / 2)
        + [ScalarType](repeating: 7, count: scalarCount / 2)
        + (scalarCount.isMultiple(of: 2) ? [] : [6])

        let tensor = TensorType(scalars)

        #expect(tensor.mean() == 6)
    }

    @Test
    func testExp() {
        let scalars: [ScalarType] = [0, 1, 2, 3, 4]
        var tensor = TensorType(scalars + [ScalarType](repeating: 0, count: ShapeType.scalarCount - scalars.count))

        let expectedScalars = scalars.map { ScalarType(Darwin.exp($0)) }
        + [ScalarType](repeating: ScalarType(Darwin.exp(Double(0))), count: ShapeType.scalarCount - scalars.count)

        tensor.exp()
        #expect(tensor.scalars == expectedScalars)
    }

    @Test
    func testSoftmax() {
        let scalars: [ScalarType] = [0, 1, -2, 3, 4]
        var tensor = TensorType(scalars + [ScalarType](repeating: 1, count: ShapeType.scalarCount - scalars.count))

        tensor.softmax()

        for scalar in tensor.scalars {
            #expect(scalar <= 1)
            #expect(scalar >= 0)
        }

        #expect(abs(tensor.sum() - 1) < 1e-6)
    }

    @Test
    func testRelu() {
        let scalars: [ScalarType] = [-1, -0.5, 0, 0.5, 1]
        let padding = [ScalarType](repeating: 0, count: ShapeType.scalarCount - scalars.count)
        var tensor = TensorType(scalars + padding)
        tensor.relu()

        let expectedScalars: [ScalarType] = [0, 0, 0, 0.5, 1] + padding
        #expect(tensor.scalars == expectedScalars)
    }
}
