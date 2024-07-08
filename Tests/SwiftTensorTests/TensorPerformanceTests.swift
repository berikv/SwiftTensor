import Foundation
import Testing
import SwiftTensor

//#if TEST_PERFORMANCE
struct TensorPerformanceTests {
    @Test
    func testCPUPerformance() {
        let r = (0..<10).map { _ in
            let startDate = Date()
            _ = runBatchCPUTensor()
            let duration = Date().timeIntervalSince(startDate)
            return duration
        }
        let mean = r.reduce(0, +) / Double(r.count)
        print("CPU duration: \(mean).")
        #expect(mean < 0.09)
        #expect(mean > 0.06)
    }

    @inline(never)
    func runBatchCPUTensor() -> Float {
        let tensorA = CPUTensor<Float, Shape784>(repeating: 1.0)
        let tensorB = CPUTensor<Float, Shape784>(repeating: 2.0)
        var result = CPUTensor<Float, Shape784>(repeating: 0.0)

        let iterations = 10000
        var sum: Float = 0
        for _ in 0..<iterations {
            result = tensorA + tensorB
            result = tensorA - tensorB
            result = tensorA * tensorB
            result = tensorA / tensorB
            let mean = result.mean()
            sum += mean
        }

        return sum
    }

    @Test
    func testSIMDPerformance() {
        let r = (0..<10).map { _ in
            let startDate = Date()
            _ = runBatchSIMDTensor()
            let duration = Date().timeIntervalSince(startDate)
            return duration
        }
        let mean = r.reduce(0, +) / Double(r.count)
        print("SIMD duration: \(mean).")
        #expect(mean < 0.055)
        #expect(mean > 0.045)
    }

    @inline(never)
    func runBatchSIMDTensor() -> Float {
        let tensorA = SIMDTensor<Float, Shape784>(repeating: 1.0)
        let tensorB = SIMDTensor<Float, Shape784>(repeating: 2.0)
        var result = SIMDTensor<Float, Shape784>(repeating: 0.0)

        let iterations = 10000
        var sum: Float = 0
        for _ in 0..<iterations {
            result = tensorA + tensorB
            result = tensorA - tensorB
            result = tensorA * tensorB
            result = tensorA / tensorB
            let mean = result.mean()
            sum += mean
        }

        return sum
    }

    @Test
    func testNeuralNetworkTraining() {
        let r = (0..<10).map { _ in
            let startDate = Date()
            _ = runNeuralNetworkTraining()
            let duration = Date().timeIntervalSince(startDate)
            return duration
        }
        let mean = r.reduce(0, +) / Double(r.count)
        print("SIMD duration: \(mean).")
        #expect(mean < 0.055)
        #expect(mean > 0.045)
    }

    struct Shape1x784: Shape {
        static var dimensionSizes: [Int] = [1, 784]
    }

    struct Shape784x128: Shape {
        static var dimensionSizes: [Int] = [784, 128]
    }

    struct Shape128x10: Shape {
        static var dimensionSizes: [Int] = [128, 10]
    }

    struct Shape1x128: Shape {
        static var dimensionSizes: [Int] = [1, 128]
    }

    struct Shape1x10: Shape {
        static var dimensionSizes: [Int] = [1, 10]
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
        let input = InputTensor(repeating: 1.0)
        var hiddenWeights = HiddenWeightsTensor.random(in: 0.0..<1.0)
        var outputWeights = OutputWeightsTensor.random(in: 0.0..<1.0)
        var hiddenBias = HiddenBiasTensor(repeating: 0.0)
        var outputBias = OutputBiasTensor(repeating: 0.0)

        var hiddenLayerOutput = HiddenLayerOutputTensor(repeating: 0.0)
        var outputLayerOutput = OutputLayerOutputTensor(repeating: 0.0)
        var error = ErrorTensor(repeating: 0.0)
        var dOutput = ErrorTensor(repeating: 0.0)
        var dHidden = HiddenLayerOutputTensor(repeating: 0.0)

        let iterations = 1000
        var loss: Float = 0

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
            loss += error.mean()

            // Backward pass
            // Output layer gradients
            dOutput = error
            var outputWeightsUpdate = OutputWeightsTensor(repeating: 0.0)
            SIMDTensor.matrixMultiplying(into: &outputWeightsUpdate, hiddenLayerOutput.transposed(), dOutput)
            SIMDTensor.adding(into: &outputWeights, outputWeights, outputWeightsUpdate)
            SIMDTensor.adding(into: &outputBias, outputBias, dOutput)

            // Hidden layer gradients
            var dHiddenRaw = HiddenLayerOutputTensor(repeating: 0.0)
            SIMDTensor.matrixMultiplying(into: &dHiddenRaw, dOutput, outputWeights.transposed())
            dHidden = dHiddenRaw.applyingReLu()

            var hiddenWeightsUpdate = HiddenWeightsTensor(repeating: 0.0)
            SIMDTensor.matrixMultiplying(into: &hiddenWeightsUpdate, input.transposed(), dHidden)
            SIMDTensor.adding(into: &hiddenWeights, hiddenWeights, hiddenWeightsUpdate)
            SIMDTensor.adding(into: &hiddenBias, hiddenBias, dHidden)
        }


        return loss / Float(iterations)
    }
}

//#endif
