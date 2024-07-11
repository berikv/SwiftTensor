
import Foundation
import SwiftTensor

testSIMDPerformance()
testNeuralNetworkTraining()

func testSIMDPerformance() {
    let measurement = Measure(1000, name: "Math") {
        runBatchMath()
    }
    
    print(measurement)
}

func testNeuralNetworkTraining() {
    let measurement = Measure(name: "Neural network training") {
        runNeuralNetworkTraining()
    }

    print(measurement)
    print(measurement.results)
}

struct Measure<R> {
    let name: String?
    let results: [R]
    let durations: [TimeInterval]
    let mean: Double
    let standardDeviation: Double

    init(_ iterations: Int = 10, name: String? = nil, _ block: () -> R) {
        precondition(iterations > 0, "Must run at least one iteration")
        
        var results = [R]()
        results.reserveCapacity(iterations)
        
        var durations = [TimeInterval]()
        durations.reserveCapacity(iterations)

        for _ in 0..<iterations {
            let startDate = Date()
            let result = block()
            let duration = Date().timeIntervalSince(startDate)
            results.append(result)
            durations.append(duration)
        }

        let mean = Double(durations.reduce(0, +)) / Double(iterations)
        let sumOfDifferences = durations.map { $0 - mean }.reduce(0, +)
        let standardDeviation = sqrt(sumOfDifferences * sumOfDifferences / Double(iterations))
        self.name = name
        self.results = results
        self.durations = durations
        self.mean = mean
        self.standardDeviation = standardDeviation
    }
}

extension Measure: CustomStringConvertible {
    var description: String {
        let nameDescription = name.map { "name: \($0), " } ?? ""
        return "\(nameDescription)"
        + "mean: \(String(format: "%.4f", mean * 1000))ms, "
        + "standardDeviation: \(String(format: "%.4e", standardDeviation))"
    }
}

struct Shape784: Shape {
    static var dimensionSizes: [Int] { [784] }
}


@inline(never)
func runBatchMath() -> Float {
    typealias T = Tensor<Shape784>
    let tensorA = T(repeating: 1.0)
    let tensorB = T(repeating: 2.0)
    var result = T(repeating: 0.0)

    let iterations = 10000
    var sum: Float = 0
    for _ in 0..<iterations {
        Tensor.adding(into: &result, tensorA, tensorB)
        Tensor.subtracting(into: &result, tensorA, tensorB)
        Tensor.multiplying(into: &result, tensorA, tensorB)
        Tensor.dividing(into: &result, tensorA, tensorB)

        let mean = result.mean()
        sum += mean
    }

    return sum
}

struct Shape1x784: Shape {
    static let dimensionSizes: [Int] = [1, 784]
}

struct Shape784x128: Shape {
    static let dimensionSizes: [Int] = [784, 128]
}

struct Shape128x10: Shape {
    static let dimensionSizes: [Int] = [128, 10]
}

struct Shape1x128: Shape {
    static let dimensionSizes: [Int] = [1, 128]
}

struct Shape1x10: Shape {
    static let dimensionSizes: [Int] = [1, 10]
}

@inline(never)
func runNeuralNetworkTraining() -> Float {
    typealias Scalar = Float

    typealias InputTensor = Tensor<Shape1x784>
    typealias HiddenWeightsTensor = Tensor<Shape784x128>
    typealias OutputWeightsTensor = Tensor<Shape128x10>
    typealias HiddenBiasTensor = Tensor<Shape1x128>
    typealias OutputBiasTensor = Tensor<Shape1x10>
    typealias HiddenLayerOutputTensor = Tensor<Shape1x128>
    typealias OutputLayerOutputTensor = Tensor<Shape1x10>
    typealias ErrorTensor = Tensor<Shape1x10>

    // Initialize tensors
    let input = InputTensor.random(in: 0.0..<1.0)
    var hiddenWeights = HiddenWeightsTensor.random(in: -0.01..<0.01)
    var outputWeights = OutputWeightsTensor.random(in: -0.01..<0.01)
    var hiddenBias = HiddenBiasTensor.zero
    var outputBias = OutputBiasTensor.zero

    var hiddenLayerOutput = HiddenLayerOutputTensor.zero
    var outputLayerOutput = OutputLayerOutputTensor.zero
    var error = ErrorTensor.zero
    var dOutput = ErrorTensor.zero
    var dHidden = HiddenLayerOutputTensor.zero

    let iterations = 1000
    var loss: Float = 0

    var outputWeightsUpdate = OutputWeightsTensor.zero
    var dHiddenRaw = HiddenLayerOutputTensor.zero
    var hiddenWeightsUpdate = HiddenWeightsTensor.zero

    for _ in 0..<iterations {
        // Forward pass
        // Hidden layer
        Tensor.matrixMultiplying(into: &hiddenLayerOutput, input, hiddenWeights)
        hiddenLayerOutput.add(hiddenBias)
        hiddenLayerOutput.relu()

        // Output layer
        Tensor.matrixMultiplying(into: &outputLayerOutput, hiddenLayerOutput, outputWeights)
        outputLayerOutput.add(outputBias)
        outputLayerOutput.softmax()

        // Calculate error (for simplicity, assuming target is all zeros)
        Tensor.subtracting(into: &error, outputLayerOutput, OutputLayerOutputTensor(repeating: 0.0))

        let mean = error.mean()
        if mean.isFinite {
            loss += mean
        } else {
            print("loss \(loss), error.mean() -> \(error.mean()), error -> \(error.description)")
        }

        // Backward pass
        // Output layer gradients
        dOutput = error.copy()
        dOutput.clip(to: 1)

        Tensor.matrixMultiplying(into: &outputWeightsUpdate, transpose: hiddenLayerOutput, dOutput)
        outputWeights.add(outputWeightsUpdate)
        outputBias.add(dOutput)

        // Hidden layer gradients
        Tensor.matrixMultiplying(into: &dHiddenRaw, dOutput, transpose: outputWeights)
        dHiddenRaw.clip(to: 1)
        dHidden = dHiddenRaw.copy()
        dHidden.relu()

        Tensor.matrixMultiplying(into: &hiddenWeightsUpdate, transpose: input, dHidden)
        hiddenWeights.add(hiddenWeightsUpdate)
        hiddenBias.add(dHidden)

        hiddenWeights.applyL2(lambda: 0.001)
        outputWeights.applyL2(lambda: 0.001)
    }

    return loss / Float(iterations)
}
