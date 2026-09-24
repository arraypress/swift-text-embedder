//
//  TextEmbedder.swift
//  TextEmbedder
//
//  Created by David Sherlock on 2026.
//
//  Qwen3-Embedding-0.6B (Alibaba, Apache 2.0) on Core AI: a text becomes a
//  1024-number unit vector; a query carries an instruction in front of it
//  (`Instruct: …\nQuery: …`), a document does not, as the model card
//  prescribes. The transformer sits in the .aimodel at fixed lengths; the
//  tokenizer, the left padding, the last-token pooling's normalisation and
//  the choice of length are here.
//

import CoreAI
import Foundation

public final class TextEmbedder: @unchecked Sendable {

    public static let assetName = "embed-qwen3-0.6b-float32.aimodel"
    /// `vocab.json`, `merges.txt` and `config.json` beside the asset.
    public static let supportFolder = "embed-support"
    public static let dimension = 1024
    /// The lengths the asset was exported at, shortest first; a text takes the shortest that holds it.
    public static let lengths = [128, 512]
    /// The model card's default retrieval instruction.
    public static let defaultInstruction = "Given a web search query, retrieve relevant passages that answer the query"

    public let url: URL
    public let tokenizer: QwenTokenizer
    private var functions: [Int: InferenceFunction] = [:]

    public enum Error: Swift.Error, CustomStringConvertible {
        case modelNotFound(String), modelUnavailable(String), emptyText
        public var description: String {
            switch self {
            case .modelNotFound(let p): return "no model at \(p)"
            case .modelUnavailable(let m): return m
            case .emptyText: return "the text is empty"
            }
        }
    }

    public init(contentsOf asset: URL, support: URL) async throws {
        guard FileManager.default.fileExists(atPath: asset.path) else { throw Error.modelNotFound(asset.path) }
        url = asset
        do { tokenizer = try QwenTokenizer(contentsOf: support) } catch { throw Error.modelUnavailable("\(support.lastPathComponent)/: \(error)") }
        var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        options.expectFrequentReshapes = true
        let model: AIModel
        do { model = try await AIModel(contentsOf: asset, options: options) } catch {
            throw Error.modelUnavailable("could not load \(asset.lastPathComponent): \(error)")
        }
        for length in Self.lengths {
            guard let f = try model.loadFunction(named: "t\(length)") else {
                throw Error.modelUnavailable("\(asset.lastPathComponent) has no t\(length) entry point; found \(model.functionNames)")
            }
            functions[length] = f
        }
    }

    public convenience init(inFolder folder: URL) async throws {
        try await self.init(contentsOf: folder.appendingPathComponent(Self.assetName), support: folder.appendingPathComponent(Self.supportFolder))
    }

    /// The query form the model was trained on.
    public static func query(_ text: String, instruction: String = defaultInstruction) -> String { "Instruct: \(instruction)\nQuery: \(text)" }

    /// The unit vector of a text (a document, or a query already wrapped with ``query(_:instruction:)``),
    /// at the shortest exported length that holds it; longer texts keep their first tokens.
    public func embed(_ text: String) async throws -> [Float] {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw Error.emptyText }
        let needed = tokenizer.count(text)
        let length = Self.lengths.first { $0 >= needed } ?? Self.lengths.last!
        let (ids, mask) = tokenizer.encode(text, length: length)
        return try await run(ids: ids, mask: mask, length: length)
    }

    /// The unit vector for ids and mask already laid out for `length`.
    public func run(ids: [Int32], mask: [Int32], length: Int) async throws -> [Float] {
        guard let function = functions[length] else { throw Error.modelUnavailable("no entry point for length \(length)") }
        var out = NDArray(shape: [1, Self.dimension], scalarType: .float32)
        var views = InferenceFunction.MutableViews()
        views.insert(out.mutableRawView(), for: "embedding")
        do {
            _ = try await function.run(inputs: ["ids": Self.ints(ids, shape: [1, length]), "mask": Self.ints(mask, shape: [1, length])],
                                       states: InferenceFunction.MutableViews(), outputViews: consume views)
        } catch { throw Error.modelUnavailable("inference failed: \(error)") }
        return Self.normalised(Self.floats(out))
    }

    public static func normalised(_ v: [Float]) -> [Float] {
        let norm = v.reduce(0) { $0 + $1 * $1 }.squareRoot()
        return norm > 0 ? v.map { $0 / norm } : v
    }

    public static func similarity(_ a: [Float], _ b: [Float]) -> Float {
        var s: Float = 0
        for i in 0..<min(a.count, b.count) { s += a[i] * b[i] }
        return s
    }

    static func ints(_ values: [Int32], shape: [Int]) -> NDArray {
        var a = NDArray(shape: shape, scalarType: .int32)
        let view = a.mutableView(as: Int32.self)
        view.withUnsafeMutablePointer { p, _, _ in values.withUnsafeBufferPointer { p.update(from: $0.baseAddress!, count: values.count) } }
        return a
    }

    static func floats(_ array: NDArray) -> [Float] {
        let count = array.shape.reduce(1, *)
        var out = [Float](repeating: 0, count: count)
        array.view(as: Float.self).withUnsafePointer { p, _, _ in out.withUnsafeMutableBufferPointer { $0.baseAddress!.update(from: p, count: count) } }
        return out
    }
}
