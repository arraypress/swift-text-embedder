//
//  EmbeddingIndex.swift
//  TextEmbedder
//
//  Created by David Sherlock on 2026.
//
//  Passages and their vectors on disk, and the question asked of them:
//  which passages are nearest this query. One file: a small header, a JSON
//  manifest of entries (source path, the passage's range and text, the
//  source's modification date and size) and the unit vectors as raw
//  float32. Scores are cosine similarities of unit vectors.
//

import Foundation

/// Passages of text and their unit vectors, searchable by a query vector.
public struct EmbeddingIndex: Sendable {

    public struct Entry: Codable, Sendable, Equatable {
        /// The file the passage came from (or any label the caller chooses).
        public var source: String
        /// Which passage of that source, from 0.
        public var passage: Int
        /// The passage's text as embedded.
        public var text: String
        /// The source's modification date and size when it was indexed.
        public var modified: Date
        public var size: Int
        public init(source: String, passage: Int, text: String, modified: Date, size: Int) {
            self.source = source; self.passage = passage; self.text = text; self.modified = modified; self.size = size
        }
    }

    public struct Hit: Sendable {
        public let entry: Entry
        public let score: Float
    }

    public struct Manifest: Codable, Sendable {
        public var model: String
        public var dimension: Int
        public var created: Date
        public var updated: Date
        public var entries: [Entry]
    }

    public static let magic = "EMBDIDX1"

    public private(set) var manifest: Manifest
    /// `entries.count × dimension`, row-major unit vectors.
    public private(set) var vectors: [Float]

    public var entries: [Entry] { manifest.entries }
    public var count: Int { manifest.entries.count }
    public var dimension: Int { manifest.dimension }

    public init(model: String, dimension: Int) {
        manifest = Manifest(model: model, dimension: dimension, created: Date(), updated: Date(), entries: [])
        vectors = []
    }

    public enum Error: Swift.Error, CustomStringConvertible {
        case notAnIndex(String), corrupt(String), dimensionMismatch(expected: Int, got: Int)
        public var description: String {
            switch self {
            case .notAnIndex(let p): return "\(p) is not an embed index"
            case .corrupt(let m): return "index is corrupt: \(m)"
            case .dimensionMismatch(let e, let g): return "vector has \(g) values; the index holds \(e)"
            }
        }
    }

    // MARK: - Editing

    /// Adds a passage. Passages of one source are replaced together with `replace(source:)`.
    public mutating func append(_ entry: Entry, embedding: [Float]) throws {
        guard embedding.count == manifest.dimension else { throw Error.dimensionMismatch(expected: manifest.dimension, got: embedding.count) }
        manifest.entries.append(entry)
        vectors.append(contentsOf: embedding)
        manifest.updated = Date()
    }

    /// Removes every passage of `source`; returns how many.
    @discardableResult
    public mutating func remove(source: String) -> Int {
        var kept: [Entry] = []
        var keptVectors: [Float] = []
        kept.reserveCapacity(count); keptVectors.reserveCapacity(vectors.count)
        var removed = 0
        for (i, e) in manifest.entries.enumerated() {
            if e.source == source { removed += 1; continue }
            kept.append(e); keptVectors.append(contentsOf: vectors[(i * dimension)..<((i + 1) * dimension)])
        }
        if removed > 0 { manifest.entries = kept; vectors = keptVectors; manifest.updated = Date() }
        return removed
    }

    public func sources() -> [String] { Array(Set(manifest.entries.map(\.source))).sorted() }

    /// True when `source` is indexed with this modification date and size.
    public func isCurrent(source: String, modified: Date, size: Int) -> Bool {
        guard let e = manifest.entries.first(where: { $0.source == source }) else { return false }
        return e.size == size && abs(e.modified.timeIntervalSince(modified)) < 1
    }

    public func embedding(at index: Int) -> [Float] { Array(vectors[(index * dimension)..<((index + 1) * dimension)]) }

    // MARK: - Questions

    /// Entries ranked by cosine similarity to `query` (a unit vector).
    public func search(_ query: [Float], limit: Int = 10, excluding source: String? = nil) -> [Hit] {
        let d = dimension
        var hits: [Hit] = []
        hits.reserveCapacity(count)
        for (i, e) in manifest.entries.enumerated() where e.source != source {
            var s: Float = 0
            let base = i * d
            for k in 0..<d { s += vectors[base + k] * query[k] }
            hits.append(Hit(entry: e, score: s))
        }
        hits.sort { $0.score > $1.score }
        return Array(hits.prefix(limit))
    }

    /// The best hit per source, ranked — a document-level answer.
    public func searchSources(_ query: [Float], limit: Int = 10) -> [Hit] {
        var best: [String: Hit] = [:]
        for hit in search(query, limit: count) where best[hit.entry.source] == nil { best[hit.entry.source] = hit }
        return Array(best.values.sorted { $0.score > $1.score }.prefix(limit))
    }

    // MARK: - Files

    public func save(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .secondsSince1970
        let json = try encoder.encode(manifest)
        var data = Data(Self.magic.utf8)
        var length = UInt32(json.count).littleEndian
        withUnsafeBytes(of: &length) { data.append(contentsOf: $0) }
        data.append(json)
        vectors.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
    }

    public init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url)
        let head = Self.magic.utf8.count
        guard data.count >= head + 4, String(decoding: data.prefix(head), as: UTF8.self) == Self.magic else { throw Error.notAnIndex(url.path) }
        let length = Int(data[head..<(head + 4)].withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(as: UInt32.self)) })
        let jsonStart = head + 4
        guard data.count >= jsonStart + length else { throw Error.corrupt("truncated manifest") }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .secondsSince1970
        let manifest: Manifest
        do { manifest = try decoder.decode(Manifest.self, from: data[jsonStart..<(jsonStart + length)]) } catch { throw Error.corrupt("manifest: \(error)") }
        let blob = data[(jsonStart + length)...]
        let expected = manifest.entries.count * manifest.dimension
        guard blob.count == expected * MemoryLayout<Float>.size else { throw Error.corrupt("\(blob.count) bytes of vectors for \(manifest.entries.count) entries") }
        var vectors = [Float](repeating: 0, count: expected)
        _ = vectors.withUnsafeMutableBytes { blob.copyBytes(to: $0) }
        self.manifest = manifest
        self.vectors = vectors
    }
}

/// Text split into passages a model can take whole.
public enum Passages {

    /// Paragraph-first splitting: blank-line blocks are joined with blank lines until adding the
    /// next would pass `maxTokens` (as counted by `count`). A single block longer than the limit
    /// is split on its own — at sentence ends, then at whitespace — and its pieces are passages
    /// of their own, never joined with other blocks. Whitespace-only blocks are dropped.
    public static func split(_ text: String, maxTokens: Int, count: (String) -> Int) -> [String] {
        let blocks = text.components(separatedBy: "\n\n").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
        var out: [String] = []
        var current = ""
        func flush() { if !current.isEmpty { out.append(current) }; current = "" }
        for block in blocks {
            if count(block) > maxTokens { flush(); out.append(contentsOf: pieces(of: block, maxTokens: maxTokens, count: count)); continue }
            if current.isEmpty { current = block; continue }
            let joined = current + "\n\n" + block
            if count(joined) <= maxTokens { current = joined } else { flush(); current = block }
        }
        flush()
        return out
    }

    /// One over-long block: sentences (a run ending in . ? or !), any sentence past the limit
    /// hard-split at whitespace, then packed with spaces up to `maxTokens`.
    static func pieces(of block: String, maxTokens: Int, count: (String) -> Int) -> [String] {
        var units: [String] = []
        var sentence = ""
        for word in block.split(whereSeparator: { $0 == " " || $0 == "\n" }) {
            sentence += (sentence.isEmpty ? "" : " ") + word
            if word.hasSuffix(".") || word.hasSuffix("?") || word.hasSuffix("!") { units.append(sentence); sentence = "" }
        }
        if !sentence.isEmpty { units.append(sentence) }
        var runs: [String] = []
        for unit in units {
            if count(unit) <= maxTokens { runs.append(unit); continue }
            var run = ""
            for word in unit.split(separator: " ") {
                let next = run.isEmpty ? String(word) : run + " " + word
                if count(next) > maxTokens && !run.isEmpty { runs.append(run); run = String(word) } else { run = next }
            }
            if !run.isEmpty { runs.append(run) }
        }
        var out: [String] = []
        var piece = ""
        for run in runs {
            let next = piece.isEmpty ? run : piece + " " + run
            if count(next) > maxTokens && !piece.isEmpty { out.append(piece); piece = run } else { piece = next }
        }
        if !piece.isEmpty { out.append(piece) }
        return out
    }
}
