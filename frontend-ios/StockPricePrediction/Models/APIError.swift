import Foundation

enum APIError: Error {
    case unauthorized
    case unknown
    case requestFailed
    
    var description: String {
        switch self {
        case .unauthorized: return "Email ou senha incorretos"
        case .requestFailed: return "Houve uma falha ao tentar realizar a requisição"
        case .unknown: return "Ocorreu um erro desconhecido."
        }
    }
}
