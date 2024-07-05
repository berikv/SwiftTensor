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
            name: "SwiftTensor"),
        .executableTarget(
            name: "SimdTester",
            dependencies: ["SwiftTensor"]
        ),
        .testTarget(
            name: "SwiftTensorTests",
            dependencies: ["SwiftTensor"]
        ),
        .testTarget(
            name: "SwiftTensorPerformanceTests",
            dependencies: ["SwiftTensor"],
            swiftSettings: [
                .define("TEST_PERFORMANCE", .when(configuration: .release)),
            ]
        ),
    ]
)
