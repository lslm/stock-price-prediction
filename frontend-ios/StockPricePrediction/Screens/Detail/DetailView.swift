//
//  DetailView.swift
//  StockPricePrediction
//
//  Created by Lucas Santos on 30/11/24.
//

import SwiftUI

struct DetailView: View {
    let company: Company
    @StateObject var detailViewModel: DetailViewModel = .init()
    
    var body: some View {
        ScrollView {
            Image(company.ticker.lowercased())
                .resizable()
                .scaledToFit()
                .frame(width: 64, height: 64)
                .clipShape(Circle())
                .padding(.top, 16)
            
            Text(company.name)
                .font(.title2)
                .fontWeight(.bold)
            
            Text(company.ticker)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            
            if (detailViewModel.screenState == .loaded && !detailViewModel.pricePredictions.isEmpty) {
                Text("\(detailViewModel.pricePredictions[0].price.formatted(.currency(code: "USD")))")
                    .font(.largeTitle)
                    .fontWeight(.medium)
                    .padding()
            
            
                PricePredictionChartView(pricePredictions: detailViewModel.pricePredictions)
                    .padding()
            }
        }
        .onAppear() {
            detailViewModel.predict(ticker: company.ticker)
        }
    }
}

#Preview {
    DetailView(company: Company.init(id: 1, name: "Apple", ticker: "AAPL"))
}
