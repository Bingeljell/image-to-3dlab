import XCTest
@testable import Core

final class CoreTests: XCTestCase {
    func testRunningChainIsReadFromServerResponse() throws {
        let state = try StudioState.decode(Data(#"{"running":{"id":"abc","status":"running"}}"#.utf8))
        XCTAssertEqual(state.runningID, "abc")
    }
    func testIdleServerHasNoRunningID() throws {
        XCTAssertNil(try StudioState.decode(Data(#"{"running":null}"#.utf8)).runningID)
    }
    func testInvalidResponseIsRejected() {
        XCTAssertThrowsError(try StudioState.decode(Data(#"{"other":true}"#.utf8)))
    }
    func testCancellationUsesStudioMutationHeader() {
        let request = StudioRequests.cancel(base: URL(string: "http://127.0.0.1:8777")!, id: "abc")
        XCTAssertEqual(request.httpMethod, "POST")
        XCTAssertEqual(request.value(forHTTPHeaderField: "X-AssetFurnace-Request"), "1")
        XCTAssertEqual(request.url?.path, "/api/chains/abc/cancel")
    }
    func testCatalogRejectsUnrelatedServer() {
        XCTAssertFalse(StudioCatalog.isValid(Data(#"{"hello":true}"#.utf8)))
        XCTAssertTrue(StudioCatalog.isValid(Data(#"{"schema_version":1,"backends":[]}"#.utf8)))
    }
    func testInstallationNeedsPythonAndServer() {
        XCTAssertFalse(Installation.isValid(at: URL(fileURLWithPath: "/nonexistent")))
    }
}
