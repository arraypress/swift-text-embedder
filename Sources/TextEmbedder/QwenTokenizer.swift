//
//  QwenTokenizer.swift
//  TextEmbedder
//
//  Created by David Sherlock on 2026.
//
//  Qwen2's byte-level byte-pair encoding, as Hugging Face's `Qwen2Tokenizer`
//  does it: its own pre-tokenising pattern (contractions case-insensitively,
//  letters with one optional non-letter prefix, single digits, punctuation
//  runs with trailing newlines, newline runs, whitespace), bytes mapped to
//  printable Unicode, merges applied by rank, then the checkpoint's end-of-
//  text token appended and padding on the LEFT to the fixed length the
//  encoder was exported with — last-token pooling reads the final position.
//  Text is NFC-normalised first, as the checkpoint's normalizer says.
//  Read from the checkpoint's own `vocab.json` and `merges.txt`.
//

import Foundation

public struct QwenTokenizer: Sendable {

    /// `<|endoftext|>`: appended to every input and used for padding.
    public let endOfText: Int32
    private let vocab: [String: Int32]
    private let ranks: [String: Int]
    private let byteToUnicode: [Character]
    private let pattern: NSRegularExpression

    public enum TokenizerError: Error, CustomStringConvertible {
        case malformed(String)
        public var description: String { switch self { case .malformed(let m): return m } }
    }

    /// `folder` holds `vocab.json`, `merges.txt` and `config.json` (the end-of-text id).
    public init(contentsOf folder: URL) throws {
        let vocabData = try Data(contentsOf: folder.appendingPathComponent("vocab.json"))
        guard let table = try JSONSerialization.jsonObject(with: vocabData) as? [String: Int] else {
            throw TokenizerError.malformed("vocab.json is not a string → int table")
        }
        vocab = table.mapValues { Int32($0) }
        // <|endoftext|> is an added token, not a vocab.json entry: its id is in the support config.
        guard let configData = try? Data(contentsOf: folder.appendingPathComponent("config.json")),
              let config = try? JSONSerialization.jsonObject(with: configData) as? [String: Any], let pad = config["pad"] as? Int else {
            throw TokenizerError.malformed("config.json with the pad/end-of-text id is missing beside vocab.json")
        }
        endOfText = Int32(pad)
        let merges = try String(contentsOf: folder.appendingPathComponent("merges.txt"), encoding: .utf8)
        var ranks: [String: Int] = [:]
        var rank = 0
        for line in merges.split(separator: "\n", omittingEmptySubsequences: true) {
            if line.hasPrefix("#") { continue }
            ranks[String(line)] = rank; rank += 1
        }
        self.ranks = ranks
        byteToUnicode = Self.bytesToUnicode()
        pattern = try NSRegularExpression(pattern: #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#)
    }

    /// The ids of `text` followed by end-of-text, no padding.
    public func tokenIDs(_ text: String) -> [Int32] {
        var ids: [Int32] = []
        for piece in pretokenize(text) { for token in bpe(piece) { if let id = vocab[token] { ids.append(id) } } }
        ids.append(endOfText)
        return ids
    }

    /// How many tokens `text` takes, end-of-text included.
    public func count(_ text: String) -> Int { tokenIDs(text).count }

    /// Ids and mask for one text at a fixed `length`: the ids of the text's LAST `length − 1`
    /// tokens (truncation keeps the end so end-of-text survives) then end-of-text, left-padded.
    public func encode(_ text: String, length: Int) -> (ids: [Int32], mask: [Int32]) {
        var ids = tokenIDs(text)
        if ids.count > length { ids = Array(ids.prefix(length - 1)) + [endOfText] }
        let pad = length - ids.count
        return ([Int32](repeating: endOfText, count: pad) + ids, [Int32](repeating: 0, count: pad) + [Int32](repeating: 1, count: ids.count))
    }

    private func pretokenize(_ text: String) -> [String] {
        let ns = text.precomposedStringWithCanonicalMapping as NSString     // the tokenizer's NFC normalizer
        return pattern.matches(in: text, range: NSRange(location: 0, length: ns.length)).map { ns.substring(with: $0.range) }
    }

    private func bpe(_ piece: String) -> [String] {
        var word: [String] = piece.utf8.map { String(byteToUnicode[Int($0)]) }
        guard word.count > 1 else { return word }
        while true {
            var best: (rank: Int, index: Int)? = nil
            for i in 0..<(word.count - 1) {
                if let r = ranks[word[i] + " " + word[i + 1]], best == nil || r < best!.rank { best = (r, i) }
            }
            guard let (_, i) = best else { break }
            let merged = word[i] + word[i + 1]
            var next: [String] = []
            var j = 0
            while j < word.count {
                if j < word.count - 1 && word[j] == word[i] && word[j + 1] == word[i + 1] { next.append(merged); j += 2 }
                else { next.append(word[j]); j += 1 }
            }
            word = next
            if word.count == 1 { break }
        }
        return word
    }

    static func bytesToUnicode() -> [Character] {
        var bytes: [Int] = Array(33...126) + Array(161...172) + Array(174...255)
        var chars = bytes
        var n = 0
        for b in 0..<256 where !bytes.contains(b) { bytes.append(b); chars.append(256 + n); n += 1 }
        var table = [Character](repeating: " ", count: 256)
        for (b, c) in zip(bytes, chars) { table[b] = Character(UnicodeScalar(c)!) }
        return table
    }
}
