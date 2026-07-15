import UIKit
import Vision
import CoreML

struct NutritionItem: Identifiable {
    let id = UUID()
    let name: String
    let value: String
}

enum DetectorError: LocalizedError {
    case modelNotFound
    case invalidImage

    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "Could not find the nutrition detection model in the app bundle."
        case .invalidImage:
            return "The captured photo could not be processed."
        }
    }
}

final class NutritionDetector {
    nonisolated(unsafe) static let shared = try? NutritionDetector()

    private let visionModel: VNCoreMLModel

    private init() throws {
        guard let modelURL = Bundle.main.url(forResource: "best", withExtension: "mlmodelc") else {
            throw DetectorError.modelNotFound
        }
        let configuration = MLModelConfiguration()
        #if targetEnvironment(simulator)
        // The Simulator has no Neural Engine and its MPSGraph/GPU backend rejects this
        // object-detector model ("Espresso exception: MpsGraph backend validation on
        // incompatible OS"), so force CPU there. Real devices keep the default (GPU/ANE).
        configuration.computeUnits = .cpuOnly
        #endif
        let mlModel = try MLModel(contentsOf: modelURL, configuration: configuration)
        visionModel = try VNCoreMLModel(for: mlModel)
    }

    /// Detects the nutrition table in the photo, OCRs the text inside it, and
    /// parses the recognized lines into name/value nutrition facts.
    nonisolated func process(_ originalImage: UIImage) throws -> [NutritionItem] {
        try recognizeNutritionText(in: originalImage).items
    }

    /// Same detection + OCR pipeline as `process`, plus health classification via
    /// `HealthClassifier` on the same recognized lines.
    nonisolated func scan(_ originalImage: UIImage) throws -> HealthScanResult {
        let (items, lines) = try recognizeNutritionText(in: originalImage)
        guard let classifier = HealthClassifier.shared else { throw ClassifierError.modelNotFound }
        let facts = NutritionLabelParser.parse(lines: lines)
        let classification = try classifier.classify(facts)
        return HealthScanResult(items: items, classification: classification)
    }

    nonisolated private func recognizeNutritionText(in originalImage: UIImage) throws -> (items: [NutritionItem], lines: [String]) {
        let image = originalImage.fixedOrientation()

        let detections = try detectNutritionTables(in: image)

        guard let bestDetection = detections.max(by: { $0.confidence < $1.confidence }) else {
            let lines = try recognizeText(in: image)
            return (parseNutritionFacts(from: lines), lines)
        }

        guard let cgImage = image.cgImage else { throw DetectorError.invalidImage }
        let pixelSize = CGSize(width: cgImage.width, height: cgImage.height)
        let rect = paddedRect(cropRect(from: bestDetection.boundingBox, imageSize: pixelSize), in: pixelSize)

        guard let croppedCGImage = cgImage.cropping(to: rect) else {
            let lines = try recognizeText(in: image)
            return (parseNutritionFacts(from: lines), lines)
        }

        let cropped = UIImage(cgImage: croppedCGImage)
        let croppedLines = try recognizeText(in: cropped)
        let croppedItems = parseNutritionFacts(from: croppedLines)
        if !croppedItems.isEmpty { return (croppedItems, croppedLines) }

        // Fall back to whole-image OCR if the cropped region didn't yield anything parseable.
        let wholeLines = try recognizeText(in: image)
        return (parseNutritionFacts(from: wholeLines), wholeLines)
    }

    nonisolated private func detectNutritionTables(in image: UIImage) throws -> [VNRecognizedObjectObservation] {
        guard let cgImage = image.cgImage else { throw DetectorError.invalidImage }
        let request = VNCoreMLRequest(model: visionModel)
        request.imageCropAndScaleOption = .scaleFit
        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: .up)
        try handler.perform([request])
        return (request.results as? [VNRecognizedObjectObservation]) ?? []
    }

    nonisolated private func recognizeText(in image: UIImage) throws -> [String] {
        guard let cgImage = image.cgImage else { return [] }
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = false
        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: .up)
        try handler.perform([request])
        let observations = (request.results as? [VNRecognizedTextObservation]) ?? []
        return observations
            .sorted { $0.boundingBox.origin.y > $1.boundingBox.origin.y }
            .compactMap { $0.topCandidates(1).first?.string }
    }

    nonisolated private func cropRect(from boundingBox: CGRect, imageSize: CGSize) -> CGRect {
        let x = boundingBox.minX * imageSize.width
        let y = (1 - boundingBox.maxY) * imageSize.height
        let width = boundingBox.width * imageSize.width
        let height = boundingBox.height * imageSize.height
        return CGRect(x: x, y: y, width: width, height: height)
    }

    nonisolated private func paddedRect(_ rect: CGRect, in imageSize: CGSize, paddingRatio: CGFloat = 0.05) -> CGRect {
        let padX = rect.width * paddingRatio
        let padY = rect.height * paddingRatio
        let expanded = rect.insetBy(dx: -padX, dy: -padY)
        return expanded.intersection(CGRect(origin: .zero, size: imageSize)).integral
    }

    nonisolated private func parseNutritionFacts(from lines: [String]) -> [NutritionItem] {
        let pattern = #"^([A-Za-z][A-Za-z\s\-']{1,29}?)[\s:]{1,3}(<?\s*\d+(?:[.,]\d+)?)\s*(kcal|kJ|mcg|µg|mg|g|%)\b"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else {
            return []
        }

        var items: [NutritionItem] = []
        for rawLine in lines {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty else { continue }
            let range = NSRange(line.startIndex..., in: line)
            guard let match = regex.firstMatch(in: line, range: range),
                  let nameRange = Range(match.range(at: 1), in: line),
                  let valueRange = Range(match.range(at: 2), in: line),
                  let unitRange = Range(match.range(at: 3), in: line) else { continue }

            let name = line[nameRange].trimmingCharacters(in: .whitespacesAndNewlines)
            let value = line[valueRange].trimmingCharacters(in: .whitespaces)
            let unit = line[unitRange]
            items.append(NutritionItem(name: name, value: "\(value)\(unit)"))
        }
        return items
    }
}

private extension UIImage {
    /// Redraws the image so its underlying pixel buffer is upright (orientation == .up),
    /// which keeps all downstream Vision + cropping math in a single consistent coordinate space.
    func fixedOrientation() -> UIImage {
        guard imageOrientation != .up else { return self }
        UIGraphicsBeginImageContextWithOptions(size, false, scale)
        defer { UIGraphicsEndImageContext() }
        draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext() ?? self
    }
}
