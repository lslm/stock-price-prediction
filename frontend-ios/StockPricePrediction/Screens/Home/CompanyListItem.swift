//
//  CompanyListItem.swift
//  StockPricePrediction
//
//  Created by Lucas Santos on 30/11/24.
//

import SwiftUI

struct CompanyListItem: View {
    let company: Company
    
    var body: some View {
        HStack {
            Image(company.ticker.lowercased())
                .resizable()
                .frame(width: 48, height: 48)
                .clipShape(Circle())
            
            VStack(alignment: .leading, spacing: 4) {
                Text(company.name)
                    .font(.headline)
                
                Text(company.ticker)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
        }
    }
}

#Preview {
    CompanyListItem(company: Company(id: 1, name: "Apple", ticker: "AAPL"))
}
