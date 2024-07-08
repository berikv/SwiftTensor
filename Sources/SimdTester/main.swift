
import Foundation
import SwiftTensor

//testSIMDPerformance()
testNeuralNetworkTraining()

func testSIMDPerformance() {
    let measurement = Measure(name: "Batch SIMD Tensor") {
        runBatchSIMDTensor()
    }
    
    print(measurement)
}

func testNeuralNetworkTraining() {
    let measurement = Measure(name: "Batch SIMD Tensor") {
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
        + "mean: \(String(format: "%.4f", mean)), "
        + "standardDeviation: \(String(format: "%.4f", standardDeviation))"
    }
}

@inline(never)
func runBatchSIMDTensor() -> Float {
    typealias SIMD = SIMDTensor<Float, Shape784>
    let tensorA = SIMD(repeating: 1.0)
    let tensorB = SIMD(repeating: 2.0)
    var result = SIMD(repeating: 0.0)

    let iterations = 10000
    var sum: Float = 0
    for _ in 0..<iterations {
        SIMD.adding(into: &result, tensorA, tensorB)
        SIMD.subtracting(into: &result, tensorA, tensorB)
        SIMD.multiplying(into: &result, tensorA, tensorB)
        SIMD.dividing(into: &result, tensorA, tensorB)

        let mean = result.mean()
        sum += mean
    }

    return sum
}

struct Shape1x784: Shape {
    static let dimensionSizes: [Int] = [1, 784]
    public static let strides: [Int] = computeStrides()
}

struct Shape784x128: Shape {
    static let dimensionSizes: [Int] = [784, 128]
    public static let strides: [Int] = computeStrides()
}

struct Shape128x10: Shape {
    static let dimensionSizes: [Int] = [128, 10]
    public static let strides: [Int] = computeStrides()
}

struct Shape1x128: Shape {
    static let dimensionSizes: [Int] = [1, 128]
    public static let strides: [Int] = computeStrides()
}

struct Shape1x10: Shape {
    static let dimensionSizes: [Int] = [1, 10]
    public static let strides: [Int] = computeStrides()
}

@inline(never)
func runNeuralNetworkTraining() -> Float {
    typealias Scalar = Float

    typealias InputTensor = SIMDTensor<Scalar, Shape1x784>
    typealias HiddenWeightsTensor = SIMDTensor<Scalar, Shape784x128>
    typealias OutputWeightsTensor = SIMDTensor<Scalar, Shape128x10>
    typealias HiddenBiasTensor = SIMDTensor<Scalar, Shape1x128>
    typealias OutputBiasTensor = SIMDTensor<Scalar, Shape1x10>
    typealias HiddenLayerOutputTensor = SIMDTensor<Scalar, Shape1x128>
    typealias OutputLayerOutputTensor = SIMDTensor<Scalar, Shape1x10>
    typealias ErrorTensor = SIMDTensor<Scalar, Shape1x10>

    // Initialize tensors
    let input = InputTensor.random(in: 0.0..<1.0)
    var inputTransposed = TransposedTensor<InputTensor>.zero
    var hiddenWeights = HiddenWeightsTensor.random(in: 0.0..<0.01)
    var outputWeights = OutputWeightsTensor.random(in: 0.0..<0.01)
    var outputWeightsTransposed = TransposedTensor<OutputWeightsTensor>.zero
    var hiddenBias = HiddenBiasTensor.zero
    var outputBias = OutputBiasTensor.zero

    var hiddenLayerOutput = HiddenLayerOutputTensor.zero
    var hiddenLayerOutputTransposed = TransposedTensor<HiddenLayerOutputTensor>.zero
    var outputLayerOutput = OutputLayerOutputTensor.zero
    var error = ErrorTensor.zero
    var dOutput = ErrorTensor.zero
    var dHidden = HiddenLayerOutputTensor.zero

    let iterations = 1_000
    var loss: Float = 0

    var outputWeightsUpdate = OutputWeightsTensor.zero
    var dHiddenRaw = HiddenLayerOutputTensor.zero
    var hiddenWeightsUpdate = HiddenWeightsTensor.zero

    for _ in 0..<iterations {
        // Forward pass
        // Hidden layer
        SIMDTensor.matrixMultiplying(into: &hiddenLayerOutput, input, hiddenWeights)
        SIMDTensor.adding(into: &hiddenLayerOutput, hiddenLayerOutput, hiddenBias)
        hiddenLayerOutput.relu()

        // Output layer
        SIMDTensor.matrixMultiplying(into: &outputLayerOutput, hiddenLayerOutput, outputWeights)
        SIMDTensor.adding(into: &outputLayerOutput, outputLayerOutput, outputBias)
        outputLayerOutput.softmax()

        // Calculate error (for simplicity, assuming target is all zeros)
        SIMDTensor.subtracting(into: &error, outputLayerOutput, OutputLayerOutputTensor(repeating: 0.0))

        let mean = error.mean()
        if mean.isFinite {
            loss += mean
        } else {
            print("loss \(loss), error.mean() -> \(error.mean()), error -> \(error)")
        }

        // Backward pass
        // Output layer gradients
        dOutput = error
        hiddenLayerOutput.transpose(into: &hiddenLayerOutputTransposed)
        SIMDTensor.matrixMultiplying(into: &outputWeightsUpdate, hiddenLayerOutputTransposed, dOutput)
        SIMDTensor.adding(into: &outputWeights, outputWeights, outputWeightsUpdate)
        SIMDTensor.adding(into: &outputBias, outputBias, dOutput)

        // Hidden layer gradients
        outputWeights.transpose(into: &outputWeightsTransposed)
        SIMDTensor.matrixMultiplying(into: &dHiddenRaw, dOutput, outputWeightsTransposed)
        dHidden = dHiddenRaw.applyingReLu()

        input.transpose(into: &inputTransposed)
        SIMDTensor.matrixMultiplying(into: &hiddenWeightsUpdate, inputTransposed, dHidden)
        SIMDTensor.adding(into: &hiddenWeights, hiddenWeights, hiddenWeightsUpdate)
        SIMDTensor.adding(into: &hiddenBias, hiddenBias, dHidden)
    }

    return loss / Float(iterations)
}
