//
//  PricePrediction.swift
//  StockPricePrediction
//
//  Created by Lucas Santos on 01/12/24.
//

import Foundation

struct PricePrediction: Codable, Identifiable {
    let id: Int
    let price: Double
    
    func getDate() -> Date {
        Calendar.current.date(byAdding: .day, value: self.id, to: Date.now)!
    }
}

struct MockData {
    let pricePredictions: [PricePrediction] = [
        .init(id: 1, price: 100),
        .init(id: 2, price: 113),
        .init(id: 3, price: 110),
        .init(id: 4, price: 105),
        .init(id: 5, price: 120),
        .init(id: 6, price: 124),
        .init(id: 7, price: 100),
        .init(id: 8, price: 115),
        .init(id: 9, price: 100),
    ]
}
