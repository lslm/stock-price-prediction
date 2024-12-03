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
            AreaMark(x: .value("Day", $0.getDate()), y: .value("Price", $0.price))
        }
        .chartYScale(domain: [
            getAveragePrice() - getAveragePrice()*0.1,
            getAveragePrice() + getAveragePrice()*0.1
        ])
        .chartXAxis() {
            AxisMarks(values: .automatic(desiredCount: pricePredictions.count))
        }
        .chartYAxis(.hidden)
        .chartXAxis(.hidden)
        .frame(height: 150)
        .foregroundStyle(getForegroundStyle().gradient)
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
    
    func getForegroundStyle() -> Color {
        let firstPrice = pricePredictions.first!.price
        let lastPrice = pricePredictions.last!.price
        
        if firstPrice > lastPrice {
            return Color.red
        } else {
            return Color.green
        }
    }
}

#Preview {
    PricePredictionChartView(pricePredictions: MockData().pricePredictions)
}
