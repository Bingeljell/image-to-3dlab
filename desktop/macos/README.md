# AssetFurnace for macOS

This wrapper opens the installed Studio in WKWebView. It requires macOS 12 or newer and an existing AssetFurnace installation with its Python environment and engines. It does not bundle or download models. The default installation is `/Applications/AssetFurnace`; choose another folder when prompted.

Run `./build.sh` to build `build/AssetFurnace.app`, or pass an output path. Building requires Xcode command line tools. The app is ad-hoc signed for local use, not notarized. Run `swift test` for the focused core tests.

The app uses the Studio on `127.0.0.1:8777`, starting it without a browser if necessary. A running server is left running on exit, including when it was started by the app, so a background job can continue. Quit offers a choice when a chain is active. Keep or discard brush work in Studio before quitting. Files and models remain in the existing installation.

## Validation and limitations

The six focused core tests cover state decoding, the cancellation request, catalog validation, and rejecting an invalid installation. They do not exercise the native window or file panels.

Local testing against AssetFurnace 0.5.7 previously covered connecting to Studio, reopening, a Pixal3D Prop + Finish generation, library persistence, and the generation quit choices. Native GLB downloading, file selection, drag-and-drop, clipboard interactions, animation playback, and 3D orbit/zoom still need hands-on validation.

Quit protection checks generation chains and an open brush, but does not comprehensively cover standalone skin jobs or all unsaved edits. Save work before quitting. There is no automatic wrapper updater; Developer ID signing and notarization are not included.
