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
        VStack {
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
                Text("Previsão de fechamento para amanhã")
                    .foregroundStyle(.secondary)
                    .padding(.top)
                
                Text(detailViewModel.pricePredictions[0].price.formatted(.currency(code: "USD")))
                    .font(.largeTitle)
                    .fontWeight(.medium)
                    .padding(.top, 1)
            
                PricePredictionChartView(pricePredictions: detailViewModel.pricePredictions)
                    
            }
            
            List {
                Section {
                    ForEach(detailViewModel.pricePredictions) { prediction in
                        HStack {
                            Text(prediction.getDate().formatted(date: .abbreviated, time: .omitted))
                            Spacer()
                            Text("\(prediction.price.formatted(.currency(code: "USD")))")
                        }
                    }
                } header: {
                    Text("Previsão para os próximos 15 dias")
                }
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
