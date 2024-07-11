import Foundation

struct Measure<R> {
    let name: String?
    let results: [R]
    let durations: [TimeInterval]
    let mean: Double
    let standardDeviation: Double

    init(_ iterations: Int = 10, name: String? = nil, _ block: () -> R) {
        precondition(iterations > 0, "Must run at least one iteration")
        let purgeCount = 2
        let totalIterations = iterations + purgeCount * 2

        var results = [R]()
        results.reserveCapacity(totalIterations)

        var durations = [TimeInterval]()
        durations.reserveCapacity(totalIterations)


        for _ in 0 ..< totalIterations {
            let startDate = Date()
            let result = block()
            let duration = Date().timeIntervalSince(startDate)
            results.append(result)
            durations.append(duration)
        }

        let resultIndices = results.indices
            .sorted { lhs, rhs in durations[lhs] > durations[rhs] }
            .dropFirst(purgeCount)
            .dropLast(purgeCount)

        results = results.enumerated()
            .filter { (index, element) in resultIndices.contains(index) }
            .map { (index, element) in element }

        durations = durations.enumerated()
            .filter { (index, element) in resultIndices.contains(index) }
            .map { (index, element) in element }

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
