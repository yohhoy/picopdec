# picopdec
Simple PNG decoder implementation.

This project is intended for study and demonstration, not for practical use. The implementation is naive and inefficiency, so its decoding speed is VERY slow.


# Limitations
- support 8bit-'Truecolour'(RGB24) or 8bit-'Truecolour with alpha'(RGBA32) only
    - 'Grayscale' and 'Indexed-colour'(Palette) are not supported
- interlaced PNG image is not supported
- ignore all ancillary chunks in PNG format
- support DEFLATE 'dynamic Huffman codes' only ('fixed Huffman codes' is not supported)
- output [PPM(portable pixmap)][ppm] image file only
    - write RGB channels, discard alpha channel for RGBA32 image

[ppm]: https://en.wikipedia.org/wiki/Netpbm_format


# Specification
- [Portable Network Graphics (PNG) Specification](http://www.w3.org/TR/PNG)
- [RFC1950 ZLIB Compressed Data Format Specification version 3.3](https://www.rfc-editor.org/rfc/rfc1950.html)
- [RFC1951 DEFLATE Compressed Data Format Specification version 1.3](https://www.rfc-editor.org/rfc/rfc1951.html)


# License
MIT License
