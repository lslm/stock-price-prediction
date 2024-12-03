//
//  HomeView.swift
//  StockPricePrediction
//
//  Created by Lucas Santos on 30/11/24.
//

import SwiftUI

struct HomeView: View {
    let companies: [Company] = [
        .init(id: 1, name: "Apple", ticker: "AAPL"),
        .init(id: 2, name: "Google", ticker: "GOOG"),
        .init(id: 3, name: "Microsoft", ticker: "MSFT"),
        .init(id: 4, name: "Meta", ticker: "META"),
        .init(id: 5, name: "Vale", ticker: "VALE")
    ]
    
    var body: some View {
        NavigationStack {
            List {
                ForEach(companies) { company in
                    NavigationLink(value: company) {
                        CompanyListItem(company: company)
                    }
                }
            }
            .navigationTitle("Stock Predictor")
            .navigationDestination(for: Company.self) { company in
                DetailView(company: company)
            }
        }
    }
}

#Preview {
    HomeView()
}
