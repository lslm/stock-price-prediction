//
//  Company.swift
//  StockPricePrediction
//
//  Created by Lucas Santos on 30/11/24.
//

import Foundation

struct Company: Identifiable, Hashable {
    let id: Int
    let name: String
    let ticker: String
}
