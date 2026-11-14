// swift-tools-version: 5.9
import PackageDescription
let package = Package(name: "AssetFurnaceDesktop", platforms: [.macOS(.v12)], products: [.library(name: "Core", targets: ["Core"]), .executable(name: "AssetFurnace", targets: ["App"])], targets: [.target(name: "Core"), .executableTarget(name: "App", dependencies: ["Core"]), .testTarget(name: "CoreTests", dependencies: ["Core"])])
