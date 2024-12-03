import SwiftUI

class DetailViewModel: ObservableObject {
    @Published var nextPrice: String = ""
    @Published var pricePredictions: [PricePrediction] = []
    @Published var screenState: ScreenState = .loading
    
    let pricePredictionClient = PricePredictionClient()
    
    
    func predict(ticker: String) {
        screenState = .loading
        
        Task {
            let result = await pricePredictionClient.predictPrice(for: ticker)
            
            DispatchQueue.main.async {
                switch result {
                case .success(let pricePredictions):
                    print(pricePredictions)
                    self.pricePredictions = pricePredictions
                    self.screenState = .loaded
                    
                case .failure(let error):
                    self.nextPrice = "Error: \(error)"
                }
            }
        }
    }
}
