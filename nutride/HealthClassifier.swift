import CoreML

/// Result of classifying a food's nutrition facts as "sehat" / "cukup sehat" / "kurang sehat".
struct HealthClassification {
    let label: String
    let confidence: Double
    let probabilities: [String: Double]
    /// `NutrientField.rawValue`s that had no reading and were filled from `FDAReference.pessimisticDefault`.
    let missingFields: [String]
}

/// Bundles a scan's raw display items with the resolved classification, so the UI can show
/// both what was read off the label and what it means.
struct HealthScanResult {
    let items: [NutritionItem]
    let classification: HealthClassification
}

enum ClassifierError: LocalizedError {
    case modelNotFound
    case missingOutput

    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "Could not find the health classification model in the app bundle."
        case .missingOutput:
            return "The classification model did not return a result."
        }
    }
}

final class HealthClassifier {
    nonisolated(unsafe) static let shared = try? HealthClassifier()

    private static let labelFeatureName = "Healthy(0)/Unhealthy(1)/In Moderation(2)"
    private static let probabilityFeatureName = "Healthy(0)/Unhealthy(1)/In Moderation(2)Probability"
    private static let healthyLabel = "sehat"
    /// "sehat" must win outright against the combined probability of the other two classes to
    /// be reported — a weak plurality (e.g. 41% sehat / 35% cukup sehat / 24% kurang sehat)
    /// isn't good enough for a health-safety call. No equivalent floor is applied to "kurang
    /// sehat" — over-flagging as unhealthy is the safe failure mode here, under-flagging isn't.
    private static let healthyConfidenceThreshold = 0.5

    private let model: MLModel

    private init() throws {
        guard let modelURL = Bundle.main.url(forResource: "nutrisi", withExtension: "mlmodelc") else {
            throw ClassifierError.modelNotFound
        }
        let configuration = MLModelConfiguration()
        #if targetEnvironment(simulator)
        configuration.computeUnits = .cpuOnly
        #endif
        model = try MLModel(contentsOf: modelURL, configuration: configuration)
    }

    /// Classifies a food from its nutrition facts. `inputs` is keyed by `NutrientField.rawValue`
    /// (the exact training-CSV column names, e.g. "Calories", "Total_Fat_DV"); any of the 15
    /// fields may be omitted and will be filled from `FDAReference.pessimisticDefault` instead —
    /// the model itself has no tolerance for missing values (it's a logistic-regression
    /// pipeline, not a tree ensemble), so every field must resolve to a real number before
    /// prediction, and an unreadable field should default toward "assume unhealthy", not
    /// "assume average".
    nonisolated func classify(_ inputs: [String: Double]) throws -> HealthClassification {
        var featureValues: [String: MLFeatureValue] = [:]
        var missingFields: [String] = []

        for field in NutrientField.allCases {
            let provided = inputs[field.rawValue]
            let value = provided ?? FDAReference.pessimisticDefault[field] ?? 0
            if provided == nil { missingFields.append(field.rawValue) }
            featureValues[field.rawValue] = field.isInteger
                ? MLFeatureValue(int64: Int64(value.rounded()))
                : MLFeatureValue(double: value)
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: featureValues)
        let output = try model.prediction(from: provider)

        guard let rawLabel = output.featureValue(for: Self.labelFeatureName)?.stringValue else {
            throw ClassifierError.missingOutput
        }

        let rawProbabilities = output.featureValue(for: Self.probabilityFeatureName)?.dictionaryValue ?? [:]
        let probabilities = Dictionary(uniqueKeysWithValues: rawProbabilities.compactMap { key, value -> (String, Double)? in
            guard let name = key as? String else { return nil }
            return (name, value.doubleValue)
        })

        let label = resolveLabel(rawLabel: rawLabel, probabilities: probabilities)

        return HealthClassification(
            label: label,
            confidence: probabilities[label] ?? 0,
            probabilities: probabilities,
            missingFields: missingFields
        )
    }

    /// Downgrades a weak "sehat" plurality to whichever of the other two classes the
    /// remaining probability mass actually favors. See `healthyConfidenceThreshold`.
    private func resolveLabel(rawLabel: String, probabilities: [String: Double]) -> String {
        guard rawLabel == Self.healthyLabel,
              (probabilities[Self.healthyLabel] ?? 0) < Self.healthyConfidenceThreshold else {
            return rawLabel
        }
        let runnerUp = probabilities
            .filter { $0.key != Self.healthyLabel }
            .max { $0.value < $1.value }
        return runnerUp?.key ?? rawLabel
    }
}
