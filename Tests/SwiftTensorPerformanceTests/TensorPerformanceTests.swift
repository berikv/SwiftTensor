import Testing
import SwiftTensor

#if TEST_PERFORMANCE
struct TensorPerformanceTests {
    @Test
    func testMathPerformance() {
        let measurement = Measure(name: "Math") {
            runBatchMath()
        }

        print(measurement)
        #expect(measurement.mean.toBeCloseTo(0.0034, margin: 0.0005))
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

    @Test
    func testNeuralNetworkTraining() {
        let measurement = Measure(name: "Neural network training") {
            runNeuralNetworkTraining()
        }

        print(measurement)
        #expect(measurement.mean.toBeCloseTo(0.22, margin: 0.02))
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

        let iterations = 500
        var loss: Float = 0


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

            var outputWeightsUpdate = OutputWeightsTensor.zero
            Tensor.matrixMultiplying(into: &outputWeightsUpdate, transpose: hiddenLayerOutput, dOutput)
            outputWeights.add(outputWeightsUpdate)
            outputBias.add(dOutput)

            // Hidden layer gradients
            Tensor.matrixMultiplying(into: &dHidden, dOutput, transpose: outputWeights)
            dHidden.clip(to: 1)
            dHidden.relu()
            
            var hiddenWeightsUpdate = HiddenWeightsTensor.zero
            Tensor.matrixMultiplying(into: &hiddenWeightsUpdate, transpose: input, dHidden)
            hiddenWeights.add(hiddenWeightsUpdate)
            hiddenBias.add(dHidden)

            hiddenWeights.applyL2(lambda: 0.001)
            outputWeights.applyL2(lambda: 0.001)
        }

        return loss / Float(iterations)
    }
}
#endif
