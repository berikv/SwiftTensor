
import Foundation
import SwiftTensor

testSIMDPerformance()

func testSIMDPerformance() {
    let r = (0..<10).map { _ in
        let startDate = Date()
        _ = runBatchSIMDTensor()
        let duration = Date().timeIntervalSince(startDate)
        return duration
    }
    let mean = r.reduce(0, +) / Double(r.count)
    print("SIMD duration: \(mean).")
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
