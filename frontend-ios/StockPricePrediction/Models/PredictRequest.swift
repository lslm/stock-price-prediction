import Foundation

struct PredictRequest: Codable {
    let ticker: String
    let daysToPredict: Int
    
    enum CodingKeys: String, CodingKey {
        case ticker
        case daysToPredict = "days_to_predict"
    }
}
