import Testing
import SwiftTensor
import Darwin

// Test edge cases using an oddly shaped Tensor
struct Shape15: Shape {
    static var dimensionSizes = [15]
}

struct CPUTensor_IntShape784_Tests {
    typealias ShapeType = Shape784
    typealias ScalarType = Int
    typealias TensorType = CPUTensor<ScalarType, ShapeType>

    @Test
    func testInitWithScalars() {
        let scalars = (0..<TensorType.ShapeType.scalarCount).map { ScalarType($0) }
        let tensor = TensorType(scalars)
        #expect(tensor.scalars.count == TensorType.ShapeType.scalarCount)
    }

    @Test
    func testInitWithRepeatingValue() {
        let value: ScalarType = ScalarType(3.14)
        let tensor = TensorType(repeating: value)
        #expect(tensor.scalars == [ScalarType](repeating: value, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testInitWithRandomRange() {
        let tensor = TensorType.random(in: 0..<1)
        let inRange = tensor.scalars.allSatisfy { $0 >= 0 && $0 <= 1 }
        #expect(inRange)
    }

    @Test
    func testSubscriptGet() {
        let tensor = TensorType([2, 3, 4] + Array<ScalarType>(repeating: 0, count: TensorType.ShapeType.scalarCount - 3))
        #expect(tensor[0] == 2)
        #expect(tensor[1] == 3)
        #expect(tensor[2] == 4)
    }

    @Test
    func testSubscriptSet() {
        var tensor = TensorType([2, 3, 4] + Array<ScalarType>(repeating: 0, count: TensorType.ShapeType.scalarCount - 3))
        tensor[3] = 8
        #expect(tensor[3] == 8)
    }

    @Test
    func testAddition() {
        let tensor1 = TensorType(repeating: 1)
        let tensor2 = TensorType(repeating: 2)
        let result = tensor1 + tensor2
        #expect(result.scalars == [ScalarType](repeating: 3, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testSubtraction() {
        let tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        let result = tensor1 - tensor2
        #expect(result.scalars == [ScalarType](repeating: 1, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testMultiplication() {
        let tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        let result = tensor1 * tensor2
        #expect(result.scalars == [ScalarType](repeating: 6, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testDivision() {
        let tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        let result = tensor1 / tensor2
        #expect(result.scalars == [ScalarType](repeating: 1, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testMin() {
        var tensor = TensorType(repeating: 10)
        let someIndex = 5
        tensor[someIndex] = 5
        #expect(tensor.min() == 5)
    }

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
        let tensor = TensorType(scalars + [ScalarType](repeating: 0, count: TensorType.ShapeType.scalarCount - scalars.count))
        let result = tensor.sum()
        #expect(result == scalars.reduce(0, +))
    }

    @Test
    func testMean() {
        let scalarCount = TensorType.ShapeType.scalarCount
        let scalars = [ScalarType](repeating: 5, count: scalarCount / 2)
        + [ScalarType](repeating: 7, count: scalarCount / 2)
        + (scalarCount.isMultiple(of: 2) ? [] : [6])

        let tensor = TensorType(scalars)

        #expect(tensor.mean() == 6)
    }
}

struct CPUTensor_Float16Shape15_Tests {
    typealias ShapeType = Shape15
    typealias ScalarType = Float16
    typealias TensorType = CPUTensor<ScalarType, ShapeType>

    @Test
    func testInitWithScalars() {
        let scalars = (0..<TensorType.ShapeType.scalarCount).map { ScalarType($0) }
        let tensor = TensorType(scalars)
        #expect(tensor.scalars.count == TensorType.ShapeType.scalarCount)
    }

    @Test
    func testInitWithRepeatingValue() {
        let value: ScalarType = ScalarType(3.14)
        let tensor = TensorType(repeating: value)
        #expect(tensor.scalars == [ScalarType](repeating: value, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testInitWithRandomRange() {
        let tensor = TensorType.random(in: 0..<1)
        let inRange = tensor.scalars.allSatisfy { $0 >= 0 && $0 <= 1 }
        #expect(inRange)
    }

    @Test
    func testSubscriptGet() {
        let tensor = TensorType([2, 3, 4] + Array<ScalarType>(repeating: 0, count: TensorType.ShapeType.scalarCount - 3))
        #expect(tensor[0] == 2)
        #expect(tensor[1] == 3)
        #expect(tensor[2] == 4)
    }

    @Test
    func testSubscriptSet() {
        var tensor = TensorType([2, 3, 4] + Array<ScalarType>(repeating: 0, count: TensorType.ShapeType.scalarCount - 3))
        tensor[3] = 8
        #expect(tensor[3] == 8)
    }

    @Test
    func testAddition() {
        let tensor1 = TensorType(repeating: 1)
        let tensor2 = TensorType(repeating: 2)
        let result = tensor1 + tensor2
        #expect(result.scalars == [ScalarType](repeating: 3, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testSubtraction() {
        let tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        let result = tensor1 - tensor2
        #expect(result.scalars == [ScalarType](repeating: 1, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testMultiplication() {
        let tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        let result = tensor1 * tensor2
        #expect(result.scalars == [ScalarType](repeating: 6, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testDivision() {
        let tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        let result = tensor1 / tensor2
        #expect(result.scalars == [ScalarType](repeating: 1.5, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testMin() {
        var tensor = TensorType(repeating: 10)
        let someIndex = 5
        tensor[someIndex] = 5
        #expect(tensor.min() == 5)
    }

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
        let tensor = TensorType(scalars + [ScalarType](repeating: 0, count: TensorType.ShapeType.scalarCount - scalars.count))
        let result = tensor.sum()
        #expect(result == scalars.reduce(0, +))
    }

    @Test
    func testMean() {
        let scalarCount = TensorType.ShapeType.scalarCount
        let scalars = [ScalarType](repeating: 5, count: scalarCount / 2)
        + [ScalarType](repeating: 7, count: scalarCount / 2)
        + (scalarCount.isMultiple(of: 2) ? [] : [6])

        let tensor = TensorType(scalars)

        #expect(tensor.mean() == 6)
    }

    @Test
    func testRelu() {
        let scalars: [ScalarType] = [-1, -0.5, 0, 0.5, 1]
        let padding = [ScalarType](repeating: 0, count: TensorType.ShapeType.scalarCount - scalars.count)
        let tensor = TensorType(scalars + padding)
        let result = tensor.relu()

        let expectedScalars: [ScalarType] = [0, 0, 0, 0.5, 1] + padding
        #expect(result.scalars == expectedScalars)
    }
}

struct CPUTensor_DoubleShape15_Tests {
    typealias ShapeType = Shape15
    typealias ScalarType = Double
    typealias TensorType = CPUTensor<ScalarType, ShapeType>

    @Test
    func testInitWithScalars() {
        let scalars = (0..<TensorType.ShapeType.scalarCount).map { ScalarType($0) }
        let tensor = TensorType(scalars)
        #expect(tensor.scalars.count == TensorType.ShapeType.scalarCount)
    }

    @Test
    func testInitWithRepeatingValue() {
        let value: ScalarType = ScalarType(3.14)
        let tensor = TensorType(repeating: value)
        #expect(tensor.scalars == [ScalarType](repeating: value, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testInitWithRandomRange() {
        let tensor = TensorType.random(in: 0..<1)
        let inRange = tensor.scalars.allSatisfy { $0 >= 0 && $0 <= 1 }
        #expect(inRange)
    }

    @Test
    func testSubscriptGet() {
        let tensor = TensorType([2, 3, 4] + Array<ScalarType>(repeating: 0, count: TensorType.ShapeType.scalarCount - 3))
        #expect(tensor[0] == 2)
        #expect(tensor[1] == 3)
        #expect(tensor[2] == 4)
    }

    @Test
    func testSubscriptSet() {
        var tensor = TensorType([2, 3, 4] + Array<ScalarType>(repeating: 0, count: TensorType.ShapeType.scalarCount - 3))
        tensor[3] = 8
        #expect(tensor[3] == 8)
    }

    @Test
    func testAddition() {
        let tensor1 = TensorType(repeating: 1)
        let tensor2 = TensorType(repeating: 2)
        let result = tensor1 + tensor2
        #expect(result.scalars == [ScalarType](repeating: 3, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testSubtraction() {
        let tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        let result = tensor1 - tensor2
        #expect(result.scalars == [ScalarType](repeating: 1, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testMultiplication() {
        let tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        let result = tensor1 * tensor2
        #expect(result.scalars == [ScalarType](repeating: 6, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testDivision() {
        let tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        let result = tensor1 / tensor2
        #expect(result.scalars == [ScalarType](repeating: 1.5, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testMin() {
        var tensor = TensorType(repeating: 10)
        let someIndex = 5
        tensor[someIndex] = 5
        #expect(tensor.min() == 5)
    }

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
        let tensor = TensorType(scalars + [ScalarType](repeating: 0, count: TensorType.ShapeType.scalarCount - scalars.count))
        let result = tensor.sum()
        #expect(result == scalars.reduce(0, +))
    }

    @Test
    func testMean() {
        let scalarCount = TensorType.ShapeType.scalarCount
        let scalars = [ScalarType](repeating: 5, count: scalarCount / 2)
        + [ScalarType](repeating: 7, count: scalarCount / 2)
        + (scalarCount.isMultiple(of: 2) ? [] : [6])

        let tensor = TensorType(scalars)

        #expect(tensor.mean() == 6)
    }

    @Test
    func testExp() {
        let scalars: [ScalarType] = [0, 1, 2, 3, 4]
        let tensor = TensorType(scalars + [ScalarType](repeating: 0, count: ShapeType.scalarCount - scalars.count))

        let expectedScalars = scalars.map { ScalarType(Darwin.exp($0)) }
        + [ScalarType](repeating: ScalarType(Darwin.exp(Double(0))), count: ShapeType.scalarCount - scalars.count)

        let result = tensor.exp()
        #expect(result.scalars == expectedScalars)
    }

    @Test
    func testRelu() {
        let scalars: [ScalarType] = [-1, -0.5, 0, 0.5, 1]
        let padding = [ScalarType](repeating: 0, count: TensorType.ShapeType.scalarCount - scalars.count)
        let tensor = TensorType(scalars + padding)
        let result = tensor.relu()

        let expectedScalars: [ScalarType] = [0, 0, 0, 0.5, 1] + padding
        #expect(result.scalars == expectedScalars)
    }

    @Test
    func testSoftmax() {
        let scalars: [ScalarType] = [0, 1, -2, 3, 4]
        let tensor = TensorType(scalars + [ScalarType](repeating: 1, count: ShapeType.scalarCount - scalars.count))

        let expectedScalars = [
            0.009458937217879949, 0.025712057155898024, 0.0012801279474991202, 0.18998783274384176,
            0.5164404733759014, 0.025712057155898024, 0.025712057155898024, 0.025712057155898024,
            0.025712057155898024, 0.025712057155898024, 0.025712057155898024, 0.025712057155898024,
            0.025712057155898024, 0.025712057155898024, 0.025712057155898024]

        let result = tensor.softmax()
        #expect(result.scalars.isApproximatelyEqual(to: expectedScalars))
    }
}

struct SIMDTensor_IntShape784_Tests {
    typealias ShapeType = Shape784
    typealias ScalarType = Int
    typealias TensorType = SIMDTensor<ScalarType, ShapeType>

    @Test
    func testInitWithScalars() {
        let scalars = (0..<TensorType.ShapeType.scalarCount).map { ScalarType($0) }
        let tensor = TensorType(scalars)
        #expect(tensor.scalars.count == TensorType.ShapeType.scalarCount)
    }

    @Test
    func testInitWithRepeatingValue() {
        let value: ScalarType = ScalarType(3.14)
        let tensor = TensorType(repeating: value)
        #expect(tensor.scalars == [ScalarType](repeating: value, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testInitWithRandomRange() {
        let tensor = TensorType.random(in: 0..<1)
        let inRange = tensor.scalars.allSatisfy { $0 >= 0 && $0 <= 1 }
        #expect(inRange)
    }

    @Test
    func testSubscriptGet() {
        let tensor = TensorType([2, 3, 4] + Array<ScalarType>(repeating: 0, count: TensorType.ShapeType.scalarCount - 3))
        #expect(tensor[0] == 2)
        #expect(tensor[1] == 3)
        #expect(tensor[2] == 4)
    }

    @Test
    func testSubscriptSet() {
        var tensor = TensorType([2, 3, 4] + Array<ScalarType>(repeating: 0, count: TensorType.ShapeType.scalarCount - 3))
        tensor[3] = 8
        #expect(tensor[3] == 8)
    }

    @Test
    func testAddition() {
        let tensor1 = TensorType(repeating: 1)
        let tensor2 = TensorType(repeating: 2)
        let result = tensor1 + tensor2
        #expect(result.scalars == [ScalarType](repeating: 3, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testSubtraction() {
        let tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        let result = tensor1 - tensor2
        #expect(result.scalars == [ScalarType](repeating: 1, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testMultiplication() {
        let tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        let result = tensor1 * tensor2
        #expect(result.scalars == [ScalarType](repeating: 6, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testDivision() {
        let tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        let result = tensor1 / tensor2
        #expect(result.scalars == [ScalarType](repeating: 1, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testMin() {
        var tensor = TensorType(repeating: 10)
        let someIndex = 5
        tensor[someIndex] = 5
        #expect(tensor.min() == 5)
    }

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
        let tensor = TensorType(scalars + [ScalarType](repeating: 0, count: TensorType.ShapeType.scalarCount - scalars.count))
        let result = tensor.sum()
        #expect(result == scalars.reduce(0, +))
    }

    @Test
    func testMean() {
        let scalarCount = TensorType.ShapeType.scalarCount
        let scalars = [ScalarType](repeating: 5, count: scalarCount / 2)
        + [ScalarType](repeating: 7, count: scalarCount / 2)
        + (scalarCount.isMultiple(of: 2) ? [] : [6])

        let tensor = TensorType(scalars)

        #expect(tensor.mean() == 6)
    }
}

struct SIMDTensor_Float16Shape15_Tests {
    typealias ShapeType = Shape15
    typealias ScalarType = Float16
    typealias TensorType = SIMDTensor<ScalarType, ShapeType>

    @Test
    func testInitWithScalars() {
        let scalars = (0..<TensorType.ShapeType.scalarCount).map { ScalarType($0) }
        let tensor = TensorType(scalars)
        #expect(tensor.scalars.count == TensorType.ShapeType.scalarCount)
    }

    @Test
    func testInitWithRepeatingValue() {
        let value: ScalarType = ScalarType(3.14)
        let tensor = TensorType(repeating: value)
        #expect(tensor.scalars == [ScalarType](repeating: value, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testInitWithRandomRange() {
        let tensor = TensorType.random(in: 0..<1)
        let inRange = tensor.scalars.allSatisfy { $0 >= 0 && $0 <= 1 }
        #expect(inRange)
    }

    @Test
    func testSubscriptGet() {
        let tensor = TensorType([2, 3, 4] + Array<ScalarType>(repeating: 0, count: TensorType.ShapeType.scalarCount - 3))
        #expect(tensor[0] == 2)
        #expect(tensor[1] == 3)
        #expect(tensor[2] == 4)
    }

    @Test
    func testSubscriptSet() {
        var tensor = TensorType([2, 3, 4] + Array<ScalarType>(repeating: 0, count: TensorType.ShapeType.scalarCount - 3))
        tensor[3] = 8
        #expect(tensor[3] == 8)
    }

    @Test
    func testAddition() {
        let tensor1 = TensorType(repeating: 1)
        let tensor2 = TensorType(repeating: 2)
        let result = tensor1 + tensor2
        #expect(result.scalars == [ScalarType](repeating: 3, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testSubtraction() {
        let tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        let result = tensor1 - tensor2
        #expect(result.scalars == [ScalarType](repeating: 1, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testMultiplication() {
        let tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        let result = tensor1 * tensor2
        #expect(result.scalars == [ScalarType](repeating: 6, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testDivision() {
        let tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        let result = tensor1 / tensor2
        #expect(result.scalars == [ScalarType](repeating: 1.5, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testMin() {
        var tensor = TensorType(repeating: 10)
        let someIndex = 5
        tensor[someIndex] = 5
        #expect(tensor.min() == 5)
    }

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
        let tensor = TensorType(scalars + [ScalarType](repeating: 0, count: TensorType.ShapeType.scalarCount - scalars.count))
        let result = tensor.sum()
        #expect(result == scalars.reduce(0, +))
    }

    @Test
    func testMean() {
        let scalarCount = TensorType.ShapeType.scalarCount
        let scalars = [ScalarType](repeating: 5, count: scalarCount / 2)
        + [ScalarType](repeating: 7, count: scalarCount / 2)
        + (scalarCount.isMultiple(of: 2) ? [] : [6])

        let tensor = TensorType(scalars)

        #expect(tensor.mean() == 6)
    }

    @Test
    func testRelu() {
        let scalars: [ScalarType] = [-1, -0.5, 0, 0.5, 1]
        let padding = [ScalarType](repeating: 0, count: TensorType.ShapeType.scalarCount - scalars.count)
        let tensor = TensorType(scalars + padding)
        let result = tensor.relu()

        let expectedScalars: [ScalarType] = [0, 0, 0, 0.5, 1] + padding
        #expect(result.scalars == expectedScalars)
    }
}

struct SIMDTensor_DoubleShape15_Tests {
    typealias ShapeType = Shape15
    typealias ScalarType = Double
    typealias TensorType = SIMDTensor<ScalarType, ShapeType>

    @Test
    func testInitWithScalars() {
        let scalars = (0..<TensorType.ShapeType.scalarCount).map { ScalarType($0) }
        let tensor = TensorType(scalars)
        #expect(tensor.scalars.count == TensorType.ShapeType.scalarCount)
    }

    @Test
    func testInitWithRepeatingValue() {
        let value: ScalarType = ScalarType(3.14)
        let tensor = TensorType(repeating: value)
        #expect(tensor.scalars == [ScalarType](repeating: value, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testInitWithRandomRange() {
        let tensor = TensorType.random(in: 0..<1)
        let inRange = tensor.scalars.allSatisfy { $0 >= 0 && $0 <= 1 }
        #expect(inRange)
    }

    @Test
    func testSubscriptGet() {
        let tensor = TensorType([2, 3, 4] + Array<ScalarType>(repeating: 0, count: TensorType.ShapeType.scalarCount - 3))
        #expect(tensor[0] == 2)
        #expect(tensor[1] == 3)
        #expect(tensor[2] == 4)
    }

    @Test
    func testSubscriptSet() {
        var tensor = TensorType([2, 3, 4] + Array<ScalarType>(repeating: 0, count: TensorType.ShapeType.scalarCount - 3))
        tensor[3] = 8
        #expect(tensor[3] == 8)
    }

    @Test
    func testAddition() {
        let tensor1 = TensorType(repeating: 1)
        let tensor2 = TensorType(repeating: 2)
        let result = tensor1 + tensor2
        #expect(result.scalars == [ScalarType](repeating: 3, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testSubtraction() {
        let tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        let result = tensor1 - tensor2
        #expect(result.scalars == [ScalarType](repeating: 1, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testMultiplication() {
        let tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        let result = tensor1 * tensor2
        #expect(result.scalars == [ScalarType](repeating: 6, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testDivision() {
        let tensor1 = TensorType(repeating: 3)
        let tensor2 = TensorType(repeating: 2)
        let result = tensor1 / tensor2
        #expect(result.scalars == [ScalarType](repeating: 1.5, count: TensorType.ShapeType.scalarCount))
    }

    @Test
    func testMin() {
        var tensor = TensorType(repeating: 10)
        let someIndex = 5
        tensor[someIndex] = 5
        #expect(tensor.min() == 5)
    }

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
        let tensor = TensorType(scalars + [ScalarType](repeating: 0, count: TensorType.ShapeType.scalarCount - scalars.count))
        let result = tensor.sum()
        #expect(result == scalars.reduce(0, +))
    }

    @Test
    func testMean() {
        let scalarCount = TensorType.ShapeType.scalarCount
        let scalars = [ScalarType](repeating: 5, count: scalarCount / 2)
        + [ScalarType](repeating: 7, count: scalarCount / 2)
        + (scalarCount.isMultiple(of: 2) ? [] : [6])

        let tensor = TensorType(scalars)

        #expect(tensor.mean() == 6)
    }

    @Test
    func testExp() {
        let scalars: [ScalarType] = [0, 1, 2, 3, 4]
        let tensor = TensorType(scalars + [ScalarType](repeating: 0, count: ShapeType.scalarCount - scalars.count))

        let expectedScalars = scalars.map { ScalarType(Darwin.exp($0)) }
        + [ScalarType](repeating: ScalarType(Darwin.exp(Double(0))), count: ShapeType.scalarCount - scalars.count)

        let result = tensor.exp()
        #expect(result.scalars == expectedScalars)
    }

    @Test
    func testSoftmax() {
        let scalars: [ScalarType] = [0, 1, -2, 3, 4]
        let tensor = TensorType(scalars + [ScalarType](repeating: 1, count: ShapeType.scalarCount - scalars.count))

        let expectedScalars = [
            0.009458937217879949, 0.025712057155898024, 0.0012801279474991202, 0.18998783274384176,
            0.5164404733759014, 0.025712057155898024, 0.025712057155898024, 0.025712057155898024,
            0.025712057155898024, 0.025712057155898024, 0.025712057155898024, 0.025712057155898024,
            0.025712057155898024, 0.025712057155898024, 0.025712057155898024]

        let result = tensor.softmax()
        #expect(result.scalars.isApproximatelyEqual(to: expectedScalars))
    }

    @Test
    func testRelu() {
        let scalars: [ScalarType] = [-1, -0.5, 0, 0.5, 1]
        let padding = [ScalarType](repeating: 0, count: TensorType.ShapeType.scalarCount - scalars.count)
        let tensor = TensorType(scalars + padding)
        let result = tensor.relu()

        let expectedScalars: [ScalarType] = [0, 0, 0, 0.5, 1] + padding
        #expect(result.scalars == expectedScalars)
    }
}

extension Array where Element: FloatingPoint {
    func isApproximatelyEqual(to other: [Element], tolerance: Element = Element.ulpOfOne) -> Bool {
        guard self.count == other.count else {
            return false
        }

        for (a, b) in zip(self, other) {
            if abs(a - b) > tolerance {
                return false
            }
        }

        return true
    }
}
