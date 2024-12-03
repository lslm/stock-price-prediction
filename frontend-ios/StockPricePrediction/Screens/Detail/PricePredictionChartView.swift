//
//  PricePredictionChartView.swift
//  StockPricePrediction
//
//  Created by Lucas Santos on 01/12/24.
//

import SwiftUI
import Charts

struct PricePredictionChartView: View {
    let pricePredictions: [PricePrediction]
    
    var body: some View {
        Chart(pricePredictions) {
            AreaMark(x: .value("Day", $0.id), y: .value("Price", $0.price))
        }
        .chartYScale(domain: [getAveragePrice() - getLowestPrice().price, getAveragePrice() + getHighestPrice().price])
        .chartXAxis {
            AxisMarks(values: .automatic(desiredCount: pricePredictions.count))
        }
        .frame(height: 200)
    }
    
    func getLowestPrice() -> PricePrediction {
        return pricePredictions.min(by: { $0.price < $1.price })!
    }
    
    func getHighestPrice() -> PricePrediction {
        return pricePredictions.max(by: { $0.price > $1.price })!
    }
    
    func getAveragePrice() -> Double {
        pricePredictions.reduce(0) { $0 + $1.price } / Double(pricePredictions.count)
    }
}

#Preview {
    PricePredictionChartView(pricePredictions: MockData().pricePredictions)
}
