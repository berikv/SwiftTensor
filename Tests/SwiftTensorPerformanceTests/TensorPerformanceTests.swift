import Foundation
import Testing
import SwiftTensor

#if TEST_PERFORMANCE
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
        #expect(mean < 0.055) // was 0.055
        #expect(mean > 0.050) // was 0.045
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
//            result = tensorA + tensorB
//            result = tensorA - tensorB
//            result = tensorA * tensorB
//            result = tensorA / tensorB
            SIMD.adding(into: &result, tensorA, tensorB)
            SIMD.subtracting(into: &result, tensorA, tensorB)
            SIMD.multiplying(into: &result, tensorA, tensorB)
            SIMD.dividing(into: &result, tensorA, tensorB)

            let mean = result.mean()
            sum += mean
        }

        return sum
    }
}

#endif
