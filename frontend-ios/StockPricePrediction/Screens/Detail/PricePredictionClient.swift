import Foundation

class PricePredictionClient {
    private let baseURL = "https://ebef-2804-7f0-9142-8ea8-e195-720e-d0e1-2a39.ngrok-free.app"
    
    func predictPrice(for ticker: String) async -> Result<[PricePrediction], APIError> {
        let urlString = "\(baseURL)/api/predict"
        let url = URL(string: urlString)!
        let predictRequest = PredictRequest(ticker: ticker, daysToPredict: 15)
        
        do {
            var request = URLRequest(url: url)
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpMethod = "POST"
            
            let requestBody = try JSONEncoder().encode(predictRequest)
            
            let (data, response) = try await URLSession.shared.upload(for: request, from: requestBody)
            
            guard let httpResponse = response as? HTTPURLResponse else {
                return .failure(.requestFailed)
            }
            
            guard httpResponse.statusCode == 200 else {
                return .failure(.unauthorized)
            }
            
            let jsonDecoder = JSONDecoder()
            jsonDecoder.keyDecodingStrategy = .convertFromSnakeCase
            let predictResponse = try jsonDecoder.decode(PredictResponse.self, from: data)
            
            let pricePredictions: [PricePrediction] = predictResponse.predictions.enumerated().map { (index, element) in
                PricePrediction(id: index + 1, price: element)
            }
            
            return .success(pricePredictions)
        } catch let error {
            print(error.localizedDescription)
            return .failure(.unknown)
        }
    }
}
