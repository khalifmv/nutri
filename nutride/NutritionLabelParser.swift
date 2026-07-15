import Foundation

/// Parses OCR'd nutrition-label lines into the classifier's 15 named features.
///
/// Unlike `NutritionDetector`'s generic name/value parser (built for display), this scans each
/// line for *any* known nutrient phrase plus the first %DV or mass reading on that line — which
/// also handles FDA's "Includes 8g Added Sugars 16%" phrasing for free, since the mass and the
/// nutrient name don't need to be adjacent.
///
/// Only fields that were actually read are included in the result. Missing keys mean "not found
/// on this label", not zero — callers (`HealthClassifier`) are responsible for imputation.
enum NutritionLabelParser {
    private static let aliases: [(field: NutrientField, phrases: [String])] = [
        (.calories, ["calories"]),
        (.totalFatDV, ["total fat"]),
        (.satFatDV, ["saturated fat", "sat fat"]),
        (.transFat, ["trans fat"]),
        (.cholesterolDV, ["cholesterol"]),
        (.sodiumDV, ["sodium"]),
        (.carbsDV, ["total carbohydrate", "carbohydrate"]),
        (.fiberDV, ["dietary fiber", "fiber"]),
        (.totalSugars, ["total sugars"]),
        (.addedSugarsDV, ["added sugars", "added sugar"]),
        (.protein, ["protein"]),
        (.vitaminDDV, ["vitamin d"]),
        (.calciumDV, ["calcium"]),
        (.ironDV, ["iron"]),
        (.potassiumDV, ["potassium"]),
    ]

    private static let massPattern = try! NSRegularExpression(
        pattern: #"(\d+(?:[.,]\d+)?)\s*(kcal|mcg|µg|mg|g|iu)\b"#,
        options: [.caseInsensitive]
    )
    private static let percentPattern = try! NSRegularExpression(pattern: #"(\d+(?:[.,]\d+)?)\s*%"#)
    private static let bareNumberPattern = try! NSRegularExpression(pattern: #"(\d+(?:[.,]\d+)?)"#)

    static func parse(lines: [String]) -> [String: Double] {
        var resolved: [String: Double] = [:]

        for rawLine in lines {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !line.isEmpty, let field = matchField(in: line), resolved[field.rawValue] == nil else { continue }

            if field.isInteger {
                if let percent = firstNumber(percentPattern, in: line) {
                    resolved[field.rawValue] = percent
                } else if let (amount, unit) = firstMass(in: line),
                          let percent = FDAReference.percentDV(amount: amount, unit: unit, field: field) {
                    resolved[field.rawValue] = percent
                }
            } else if let (amount, _) = firstMass(in: line) {
                resolved[field.rawValue] = amount
            } else if field == .calories, let bare = firstNumber(bareNumberPattern, in: line) {
                resolved[field.rawValue] = bare
            }
        }

        return resolved
    }

    private static func matchField(in line: String) -> NutrientField? {
        aliases.first { _, phrases in phrases.contains { line.localizedCaseInsensitiveContains($0) } }?.field
    }

    private static func firstNumber(_ regex: NSRegularExpression, in line: String) -> Double? {
        let range = NSRange(line.startIndex..., in: line)
        guard let match = regex.firstMatch(in: line, range: range),
              let numberRange = Range(match.range(at: 1), in: line) else { return nil }
        return Double(line[numberRange].replacingOccurrences(of: ",", with: "."))
    }

    private static func firstMass(in line: String) -> (amount: Double, unit: String)? {
        let range = NSRange(line.startIndex..., in: line)
        guard let match = massPattern.firstMatch(in: line, range: range),
              let numberRange = Range(match.range(at: 1), in: line),
              let unitRange = Range(match.range(at: 2), in: line),
              let value = Double(line[numberRange].replacingOccurrences(of: ",", with: ".")) else { return nil }
        return (value, String(line[unitRange]).lowercased())
    }
}
