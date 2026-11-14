import AppKit
import WebKit
import Core

@MainActor
private final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate, WKNavigationDelegate, WKUIDelegate, WKDownloadDelegate {
    private let base = URL(string: "http://127.0.0.1:8777")!
    private var window: NSWindow!
    private var web: WKWebView!
    private var server: Process?
    private var terminating = false
    private var checkingClose = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        let config = WKWebViewConfiguration()
        web = WKWebView(frame: .zero, configuration: config)
        web.navigationDelegate = self
        web.uiDelegate = self
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1280, height: 800), styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        window.title = "AssetFurnace"
        window.contentView = web
        window.delegate = self
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        makeMenu()
        Task { await connect() }
    }
    private func makeMenu() {
        let menu = NSMenu()
        let appItem = NSMenuItem(); menu.addItem(appItem)
        let appMenu = NSMenu(); appItem.submenu = appMenu
        appMenu.addItem(withTitle: "Quit AssetFurnace", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        let editItem = NSMenuItem(); menu.addItem(editItem)
        let edit = NSMenu(title: "Edit"); editItem.submenu = edit
        edit.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
        edit.addItem(.separator())
        edit.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        edit.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        edit.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        NSApp.mainMenu = menu
    }
    private func get(_ path: String) async throws -> Data {
        var request = URLRequest(url: base.appendingPathComponent(path))
        request.timeoutInterval = 3
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else { throw URLError(.badServerResponse) }
        return data
    }
    private func alive() async -> Bool {
        guard let data = try? await get("api/catalog") else { return false }
        return StudioCatalog.isValid(data)
    }
    private func connect() async {
        if await alive() { load(); return }
        let saved = UserDefaults.standard.string(forKey: "installation") ?? "/Applications/AssetFurnace"
        var root = URL(fileURLWithPath: saved, isDirectory: true)
        if !Installation.isValid(at: root) {
            let picker = NSOpenPanel(); picker.canChooseFiles = false; picker.canChooseDirectories = true
            picker.message = "Select the AssetFurnace installation folder."
            guard picker.runModal() == .OK, let selected = picker.url, Installation.isValid(at: selected) else {
                show("AssetFurnace installation not found", "Select a folder containing viewer/serve.py and .venv/bin/python, then reopen the app.")
                return
            }
            root = selected; UserDefaults.standard.set(root.path, forKey: "installation")
        }
        let process = Process()
        process.executableURL = root.appendingPathComponent(".venv/bin/python")
        process.arguments = [root.appendingPathComponent("viewer/serve.py").path, "--no-browser", "--port", "8777"]
        process.currentDirectoryURL = root
        process.standardOutput = FileHandle.nullDevice; process.standardError = FileHandle.nullDevice
        do { try process.run(); server = process } catch { show("Could not start AssetFurnace", error.localizedDescription); return }
        for _ in 0..<40 {
            try? await Task.sleep(nanoseconds: 250_000_000)
            if await alive() { load(); return }
            if !process.isRunning { break }
        }
        show("Could not start AssetFurnace", "The server did not answer on port 8777. The port may be in use, or the installation may need repair.")
    }
    private func load() { web.load(URLRequest(url: base.appendingPathComponent("viewer/studio.html"))) }
    private func show(_ title: String, _ message: String) {
        let alert = NSAlert(); alert.messageText = title; alert.informativeText = message; alert.runModal()
    }
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        window.makeKeyAndOrderFront(nil); return true
    }
    func windowShouldClose(_ sender: NSWindow) -> Bool { NSApp.terminate(nil); return false }
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        if terminating { return .terminateNow }
        guard !checkingClose else { return .terminateCancel }
        checkingClose = true
        Task { await checkClose() }
        return .terminateLater
    }
    private func checkClose() async {
        defer { checkingClose = false }
        do {
            let dirty = try await web.evaluateJavaScript("Boolean(document.querySelector('.brushstage, .sv-brush:not([hidden])'))") as? Bool ?? true
            if dirty {
                show("Unsaved edits", "Return to the screen and keep or discard the brush work before quitting.")
                NSApp.reply(toApplicationShouldTerminate: false); return
            }
            let state = try StudioState.decode(await get("api/chains"))
            if let id = state.runningID {
                let alert = NSAlert(); alert.messageText = "A production is running"
                alert.informativeText = "You can keep it running in the background, cancel it, or return to the studio."
                alert.addButton(withTitle: "Arka planda sürdür")
                alert.addButton(withTitle: "Üretimi iptal edip çık")
                alert.addButton(withTitle: "Geri dön")
                let choice = alert.runModal()
                if choice == .alertThirdButtonReturn { NSApp.reply(toApplicationShouldTerminate: false); return }
                if choice == .alertSecondButtonReturn {
                    let request = StudioRequests.cancel(base: base, id: id)
                    let (data, response) = try await URLSession.shared.data(for: request)
                    guard (response as? HTTPURLResponse)?.statusCode == 200,
                          let result = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                          result["cancelled"] as? Bool == true else { throw URLError(.badServerResponse) }
                }
            }
            terminating = true; NSApp.reply(toApplicationShouldTerminate: true)
        } catch {
            let alert = NSAlert(); alert.messageText = "Could not check production"
            alert.informativeText = "The studio did not report its state. Quitting may discard unsaved work.\n\n\(error.localizedDescription)"
            alert.addButton(withTitle: "Geri dön"); alert.addButton(withTitle: "Quit Anyway")
            let quit = alert.runModal() == .alertSecondButtonReturn
            terminating = quit; NSApp.reply(toApplicationShouldTerminate: quit)
        }
    }
    func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction, decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        if navigationAction.shouldPerformDownload { decisionHandler(.download); return }
        if let url = navigationAction.request.url, let scheme = url.scheme, ["http", "https"].contains(scheme),
           !(url.host == "127.0.0.1" && url.port == 8777) {
            NSWorkspace.shared.open(url); decisionHandler(.cancel); return
        }
        decisionHandler(.allow)
    }
    func webView(_ webView: WKWebView, decidePolicyFor navigationResponse: WKNavigationResponse, decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void) {
        decisionHandler(navigationResponse.canShowMIMEType ? .allow : .download)
    }
    func webView(_ webView: WKWebView, navigationAction: WKNavigationAction, didBecome download: WKDownload) { download.delegate = self }
    func webView(_ webView: WKWebView, navigationResponse: WKNavigationResponse, didBecome download: WKDownload) { download.delegate = self }
    func download(_ download: WKDownload, decideDestinationUsing response: URLResponse, suggestedFilename: String, completionHandler: @escaping (URL?) -> Void) {
        let panel = NSSavePanel(); panel.nameFieldStringValue = suggestedFilename
        panel.begin { completionHandler($0 == .OK ? panel.url : nil) }
    }
    func download(_ download: WKDownload, didFailWithError error: Error, resumeData: Data?) { show("Download failed", error.localizedDescription) }
    func webView(_ webView: WKWebView, runOpenPanelWith parameters: WKOpenPanelParameters, initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping ([URL]?) -> Void) {
        let panel = NSOpenPanel(); panel.allowsMultipleSelection = parameters.allowsMultipleSelection
        panel.canChooseDirectories = parameters.allowsDirectories
        panel.begin { completionHandler($0 == .OK ? panel.urls : nil) }
    }
    func webView(_ webView: WKWebView, runJavaScriptAlertPanelWithMessage message: String, initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping () -> Void) { show("AssetFurnace", message); completionHandler() }
    func webView(_ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message: String, initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping (Bool) -> Void) {
        let alert = NSAlert(); alert.messageText = message; alert.addButton(withTitle: "OK"); alert.addButton(withTitle: "Cancel")
        completionHandler(alert.runModal() == .alertFirstButtonReturn)
    }
    func webView(_ webView: WKWebView, runJavaScriptTextInputPanelWithPrompt prompt: String, defaultText: String?, initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping (String?) -> Void) {
        let alert = NSAlert(); alert.messageText = prompt
        let input = NSTextField(string: defaultText ?? ""); input.frame.size.width = 320; alert.accessoryView = input
        alert.addButton(withTitle: "OK"); alert.addButton(withTitle: "Cancel")
        completionHandler(alert.runModal() == .alertFirstButtonReturn ? input.stringValue : nil)
    }
    func webView(_ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration, for navigationAction: WKNavigationAction, windowFeatures: WKWindowFeatures) -> WKWebView? {
        if let url = navigationAction.request.url { NSWorkspace.shared.open(url) }; return nil
    }
}
MainActor.assumeIsolated {
    let application = NSApplication.shared
    let delegate = AppDelegate()
    application.delegate = delegate
    application.setActivationPolicy(.regular)
    application.run()
}
