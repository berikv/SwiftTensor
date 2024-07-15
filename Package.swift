// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "SwiftTensor",
    platforms: [
        .macOS(.v14),
        .iOS(.v16)
    ],
    products: [
        .library(
            name: "SwiftTensor",
            targets: ["SwiftTensor"]),
        .executable(
            name: "tensor-cli",
            targets: ["tensor-cli"]),
    ],
    targets: [
        .target(
            name: "SwiftTensor",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .executableTarget(
            name: "tensor-cli",
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
