
extension FloatingPoint {
    func toBeCloseTo(_ other: Self, margin: Self = .ulpOfOne) -> Bool {
        abs(self - other) < margin
    }
}
