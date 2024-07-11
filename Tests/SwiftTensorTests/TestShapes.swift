import SwiftTensor

struct Shape784: Shape {
    static var dimensionSizes: [Int] { [784] }
}

struct Shape15: Shape {
    static var dimensionSizes: [Int] { [15] }
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

typealias Tensor2x2 = Tensor<Shape2x2>
typealias Tensor2x3 = Tensor<Shape2x3>
typealias Tensor3x2 = Tensor<Shape3x2>
typealias Tensor3x3 = Tensor<Shape3x3>
typealias Tensor4x4 = Tensor<Shape4x4>

struct Shape2x2: Shape {
    public static var dimensionSizes: [Int] { [2, 2] }
}

struct Shape2x3: Shape {
    public static var dimensionSizes: [Int] { [2, 3] }
}

struct Shape3x2: Shape {
    public static var dimensionSizes: [Int] { [3, 2] }
}

struct Shape3x3: Shape {
    public static var dimensionSizes: [Int] { [3, 3] }
}

struct Shape4x4: Shape {
    public static var dimensionSizes: [Int] { [4, 4] }
}
