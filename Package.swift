// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "SwiftTensor",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "SwiftTensor",
            targets: ["SwiftTensor"]),
        .executable(
            name: "SimdTester",
            targets: ["SimdTester"]),
    ],
    targets: [
        .target(
            name: "SwiftTensor",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .executableTarget(
            name: "SimdTester",
            dependencies: ["SwiftTensor"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .testTarget(
            name: "SwiftTensorTests",
            dependencies: ["SwiftTensor"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .testTarget(
            name: "SwiftTensorPerformanceTests",
            dependencies: ["SwiftTensor"],
            swiftSettings: [
                .define("TEST_PERFORMANCE", .when(configuration: .release)),
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
    ]
)
