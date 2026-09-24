//
//  TextEmbedderTests.swift
//  TextEmbedderTests
//
//  Created by David Sherlock on 2026.
//
//  The Qwen3-Embedding port against transformers: the tokenizer on the
//  fixture texts (no model needed), then — with the asset installed under
//  ~/Library/Application Support/embed — every fixture embedding at every
//  exported length; plus the index and the passage splitter.
//

import Foundation
import Testing
@testable import TextEmbedder

private func fixtureURL(_ name: String, _ ext: String) throws -> URL {
    try #require(Bundle.module.url(forResource: name, withExtension: ext, subdirectory: "Fixtures"))
}
private func floats(_ name: String) throws -> [Float] {
    try Data(contentsOf: fixtureURL(name, "f32")).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}
private func psnr(_ reference: [Float], _ got: [Float]) -> Double {
    var err = 0.0, lo = Double.infinity, hi = -Double.infinity
    for i in reference.indices { let d = Double(reference[i]) - Double(got[i]); err += d * d; lo = min(lo, Double(reference[i])); hi = max(hi, Double(reference[i])) }
    let mse = err / Double(reference.count)
    return mse == 0 ? .infinity : 20 * log10((hi - lo) / mse.squareRoot())
}
private struct Fixture: Decodable {
    struct PerLength: Decodable { let ids: [[Int32]]; let mask: [[Int32]] }
    let instruction: String; let texts: [String]; let long: String; let per_length: [String: PerLength]; let counts: [Int]
}
private func fixture() throws -> Fixture { try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: fixtureURL("text_fixture", "json"))) }
private let installed = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support/embed")
private var modelInstalled: Bool { FileManager.default.fileExists(atPath: installed.appendingPathComponent(TextEmbedder.assetName).path) }

@Suite struct TokenizerTests {

    @Test("every fixture text tokenises to Qwen2Tokenizer's ids and mask at every length")
    func fixtureTexts() throws {
        let fx = try fixture()
        let tokenizer = try QwenTokenizer(contentsOf: fixtureURL("vocab", "json").deletingLastPathComponent())
        let all = fx.texts + [fx.long]
        for (i, text) in all.enumerated() { #expect(tokenizer.count(text) == fx.counts[i], "count of text \(i)") }
        for (key, per) in fx.per_length {
            let length = Int(key)!
            for (i, text) in all.enumerated() {
                let (ids, mask) = tokenizer.encode(text, length: length)
                #expect(ids == per.ids[i], "L=\(length) text \(i): \(Array(ids.suffix(8))) vs \(Array(per.ids[i].suffix(8)))")
                #expect(mask == per.mask[i], "L=\(length) text \(i) mask")
            }
        }
    }

    @Test("end-of-text is appended, padding is on the left, truncation keeps end-of-text")
    func layout() throws {
        let tokenizer = try QwenTokenizer(contentsOf: fixtureURL("vocab", "json").deletingLastPathComponent())
        let (ids, mask) = tokenizer.encode("Hello world", length: 8)
        #expect(ids == [151643, 151643, 151643, 151643, 151643, 9707, 1879, 151643])
        #expect(mask == [0, 0, 0, 0, 0, 1, 1, 1])
        let (long, lmask) = tokenizer.encode(String(repeating: "word ", count: 50), length: 8)
        #expect(long.count == 8 && long.last == 151643 && lmask.allSatisfy { $0 == 1 })
    }
}

@Suite struct IndexTests {

    @Test("round-trips passages and vectors, searches, and replaces a source")
    func roundTrip() throws {
        var index = EmbeddingIndex(model: "test", dimension: 3)
        let when = Date(timeIntervalSince1970: 1_700_000_000.5)
        try index.append(.init(source: "/a.md", passage: 0, text: "alpha", modified: when, size: 5), embedding: TextEmbedder.normalised([1, 0, 0]))
        try index.append(.init(source: "/a.md", passage: 1, text: "beta", modified: when, size: 5), embedding: TextEmbedder.normalised([1, 1, 0]))
        try index.append(.init(source: "/b.md", passage: 0, text: "gamma", modified: when, size: 6), embedding: TextEmbedder.normalised([0, 0, 1]))
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("embed-\(UUID().uuidString).idx")
        defer { try? FileManager.default.removeItem(at: url) }
        try index.save(to: url)
        let back = try EmbeddingIndex(contentsOf: url)
        #expect(back.entries == index.entries && back.vectors == index.vectors)
        #expect(back.search(TextEmbedder.normalised([1, 1, 0]), limit: 2).map(\.entry.text) == ["beta", "alpha"])
        #expect(back.searchSources(TextEmbedder.normalised([1, 1, 0])).map(\.entry.source) == ["/a.md", "/b.md"])
        #expect(back.isCurrent(source: "/a.md", modified: when, size: 5) && !back.isCurrent(source: "/a.md", modified: when, size: 6))
        var mutable = back
        #expect(mutable.remove(source: "/a.md") == 2 && mutable.count == 1 && mutable.vectors.count == 3)
        #expect(throws: EmbeddingIndex.Error.self) { try mutable.append(.init(source: "/c", passage: 0, text: "", modified: when, size: 0), embedding: [1, 2]) }
    }
}

@Suite struct PassageTests {

    @Test("paragraphs are joined up to the limit and long ones split at sentences")
    func splitting() {
        let count: (String) -> Int = { $0.split(separator: " ").count }
        let text = "One two three.\n\nFour five.\n\nSix seven eight nine ten eleven twelve. Thirteen fourteen."
        let passages = Passages.split(text, maxTokens: 6, count: count)
        #expect(passages == ["One two three.\n\nFour five.", "Six seven eight nine ten eleven", "twelve. Thirteen fourteen."])
        #expect(Passages.split("\n\n  \n\n", maxTokens: 6, count: count).isEmpty)
        #expect(Passages.split("a b c d e f g h i j", maxTokens: 4, count: count) == ["a b c d", "e f g h", "i j"])
    }
}

@Suite(.serialized) struct ModelTests {

    @Test("every fixture embedding matches transformers at every length", .enabled(if: modelInstalled))
    func embeddings() async throws {
        let embedder = try await TextEmbedder(inFolder: installed)
        let fx = try fixture()
        for (key, per) in fx.per_length {
            let length = Int(key)!
            let reference = try floats("embeddings_\(length)")
            var worst = Double.infinity, worstCos: Float = 1
            for i in per.ids.indices {
                let e = try await embedder.run(ids: per.ids[i], mask: per.mask[i], length: length)
                let r = Array(reference[(i * TextEmbedder.dimension)..<((i + 1) * TextEmbedder.dimension)])
                worst = min(worst, psnr(r, e)); worstCos = min(worstCos, TextEmbedder.similarity(r, e))
            }
            print("L=\(length): \(per.ids.count) texts, worst \(String(format: "%.1f", worst)) dB, worst cosine \(worstCos)")
            #expect(worstCos > 0.9999, "L=\(length)")
        }
    }

    @Test("embed(_:) picks the shortest length that holds the text and matches the fixture", .enabled(if: modelInstalled))
    func endToEnd() async throws {
        let embedder = try await TextEmbedder(inFolder: installed)
        let fx = try fixture()
        let short = try floats("embeddings_128"), long = try floats("embeddings_512")
        let e0 = try await embedder.embed(fx.texts[0])
        #expect(TextEmbedder.similarity(e0, Array(short[0..<1024])) > 0.9999)
        let eLong = try await embedder.embed(fx.long)
        let i = fx.texts.count
        #expect(TextEmbedder.similarity(eLong, Array(long[(i * 1024)..<((i + 1) * 1024)])) > 0.9999)
        // a query lands nearer its answer than an unrelated document
        let q = try await embedder.embed(TextEmbedder.query("what is the capital of France?"))
        let paris = try await embedder.embed("Paris is the capital and most populous city of France.")
        let kick = try await embedder.embed("A punchy 909-style kick sits under most techno tracks; layer a sub for weight.")
        print("cos(query, paris) \(TextEmbedder.similarity(q, paris)) cos(query, kick) \(TextEmbedder.similarity(q, kick))")
        #expect(TextEmbedder.similarity(q, paris) > TextEmbedder.similarity(q, kick) + 0.2)
    }
}
