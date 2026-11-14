import Foundation

public struct StudioState: Decodable {
    private struct Run: Decodable { let id: String }
    private let running: Run?
    private enum CodingKeys: String, CodingKey { case running }
    public var runningID: String? { running?.id }
    public static func decode(_ data: Data) throws -> StudioState {
        try JSONDecoder().decode(StudioState.self, from: data)
    }
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        guard values.contains(.running) else {
            throw DecodingError.keyNotFound(CodingKeys.running, .init(codingPath: decoder.codingPath, debugDescription: "Missing running"))
        }
        running = try values.decodeIfPresent(Run.self, forKey: .running)
    }
}
public enum Installation {
    public static func isValid(at root: URL) -> Bool {
        let fm = FileManager.default
        return fm.isExecutableFile(atPath: root.appendingPathComponent(".venv/bin/python").path)
            && fm.fileExists(atPath: root.appendingPathComponent("viewer/serve.py").path)
            && fm.fileExists(atPath: root.appendingPathComponent("viewer/studio.html").path)
    }
}
public enum StudioCatalog {
    public static func isValid(_ data: Data) -> Bool {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
        return object["schema_version"] as? Int == 1 && object["backends"] is [Any]
    }
}
public enum StudioRequests {
    public static func cancel(base: URL, id: String) -> URLRequest {
        var request = URLRequest(url: base.appendingPathComponent("api/chains/\(id)/cancel"))
        request.httpMethod = "POST"
        request.timeoutInterval = 10
        request.setValue("1", forHTTPHeaderField: "X-AssetFurnace-Request")
        return request
    }
}
