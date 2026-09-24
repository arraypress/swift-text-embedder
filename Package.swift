// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "swift-text-embedder",
    platforms: [
        .macOS("27.0"),
    ],
    products: [
        .library(name: "TextEmbedder", targets: ["TextEmbedder"]),
    ],
    targets: [
        .target(name: "TextEmbedder"),
        .testTarget(
            name: "TextEmbedderTests",
            dependencies: ["TextEmbedder"],
            resources: [.copy("Fixtures")]
        ),
    ]
)
