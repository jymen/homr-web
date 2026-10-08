// Renders one page (default 1) of a PDF to a white-backed RGB PNG with CoreGraphics (macOS only).
// usage: swiftc -O tools/rasterise-pdf.swift -o <bin> && <bin> <in.pdf> <out.png> [dpi=300] [page=1]
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

let args = CommandLine.arguments
let pageNumber = args.count > 4 ? Int(args[4]) ?? 1 : 1
guard args.count >= 3, let doc = CGPDFDocument(URL(fileURLWithPath: args[1]) as CFURL), let page = doc.page(at: pageNumber) else {
  FileHandle.standardError.write("usage: rasterise-pdf <in.pdf> <out.png> [dpi] [page]; the file must be a PDF holding that page\n".data(using: .utf8)!)
  exit(2)
}
let scale = (args.count > 3 ? Double(args[3]) ?? 300 : 300) / 72
let box = page.getBoxRect(.mediaBox)
let width = Int(box.width * scale), height = Int(box.height * scale)
let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                    space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
ctx.scaleBy(x: scale, y: scale)
ctx.translateBy(x: -box.origin.x, y: -box.origin.y)
ctx.drawPDFPage(page)
let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: args[2]) as CFURL, UTType.png.identifier as CFString, 1, nil)!
CGImageDestinationAddImage(dest, ctx.makeImage()!, nil)
exit(CGImageDestinationFinalize(dest) ? 0 : 1)
