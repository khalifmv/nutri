import Foundation

/// The 15 nutrition-label features the "nutrisi" CoreML classifier was trained on.
/// Raw values match the exact CoreML input names (and the training CSV column names),
/// so a dictionary keyed by `rawValue` can be fed straight into both.
enum NutrientField: String, CaseIterable {
    case calories = "Calories"
    case totalFatDV = "Total_Fat_DV"
    case satFatDV = "Sat_Fat_DV"
    case transFat = "Trans_Fat"
    case cholesterolDV = "Cholesterol_DV"
    case sodiumDV = "Sodium_DV"
    case carbsDV = "Carbs_DV"
    case fiberDV = "Fiber_DV"
    case totalSugars = "Total_Sugars (g)"
    case addedSugarsDV = "Added_Sugars_DV"
    case protein = "Protein (g)"
    case vitaminDDV = "Vitamin_D_DV"
    case calciumDV = "Calcium_DV"
    case ironDV = "Iron_DV"
    case potassiumDV = "Potassium_DV"

    /// These ten are %DV columns and are Int64 at the CoreML boundary; the rest are raw
    /// units (kcal/g) and are Double. Confirmed by inspecting the exported model spec.
    static let integerFields: Set<NutrientField> = [
        .totalFatDV, .satFatDV, .cholesterolDV, .sodiumDV, .carbsDV,
        .addedSugarsDV, .vitaminDDV, .calciumDV, .ironDV, .potassiumDV,
    ]

    var isInteger: Bool { Self.integerFields.contains(self) }
}

/// FDA reference values (21 CFR 101.9(c), 2016/2020 revised Nutrition Facts label) used to
/// convert a raw mass reading into %DV, plus safe fallback defaults for fields a scan can't
/// read at all.
enum FDAReference {
    /// Daily Value amount + unit for each %DV field.
    static let dailyValue: [NutrientField: (amount: Double, unit: String)] = [
        .totalFatDV: (78, "g"),
        .satFatDV: (20, "g"),
        .cholesterolDV: (300, "mg"),
        .sodiumDV: (2300, "mg"),
        .carbsDV: (275, "g"),
        .fiberDV: (28, "g"),
        .addedSugarsDV: (50, "g"),
        .vitaminDDV: (20, "mcg"),
        .calciumDV: (1300, "mg"),
        .ironDV: (18, "mg"),
        .potassiumDV: (4700, "mg"),
    ]

    /// Median of each feature *within the "kurang sehat" class only* (51 of the 221 training
    /// rows), used as a pessimistic fallback for fields a scan can't read at all.
    ///
    /// The overall-dataset median was tried first and caused real misclassifications: several
    /// %DV fields sit much lower for unhealthy foods' medians pulled toward the middle. A
    /// missing reading should read as "assume the worst until proven otherwise", not "assume
    /// average" — so this deliberately biases every unreadable field toward the unhealthy
    /// class's typical value instead of the whole dataset's.
    static let pessimisticDefault: [NutrientField: Double] = [
        .calories: 370, .totalFatDV: 26, .satFatDV: 37, .transFat: 0,
        .cholesterolDV: 10, .sodiumDV: 23, .carbsDV: 14, .fiberDV: 8,
        .totalSugars: 8.5, .addedSugarsDV: 0, .protein: 9, .vitaminDDV: 0,
        .calciumDV: 6, .ironDV: 8, .potassiumDV: 6,
    ]

    /// Converts a raw amount+unit reading (e.g. 160mg) into %DV for the given field,
    /// rounded to a whole number since the ten %DV inputs are Int64 at the CoreML boundary.
    /// Returns nil if `field` has no established %DV.
    static func percentDV(amount: Double, unit: String, field: NutrientField) -> Double? {
        guard let reference = dailyValue[field] else { return nil }
        let normalized = convert(amount: amount, from: unit.lowercased(), to: reference.unit, field: field)
        return (normalized / reference.amount * 100).rounded()
    }

    private static func convert(amount: Double, from unit: String, to target: String, field: NutrientField) -> Double {
        if unit == target { return amount }
        switch (unit, target) {
        case ("mg", "g"): return amount / 1000
        case ("g", "mg"): return amount * 1000
        case ("mcg", "mg"), ("\u{00B5}g", "mg"): return amount / 1000
        case ("mg", "mcg"), ("mg", "\u{00B5}g"): return amount * 1000
        case ("iu", "mcg") where field == .vitaminDDV: return amount / 40
        default: return amount
        }
    }
}
