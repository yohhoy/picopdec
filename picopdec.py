#!/usr/bin/env python3
"""
pico PNG decoder
Copyright (c) 2023 yohhoy
"""
from itertools import zip_longest
import struct
import sys

#TRACE = print
TRACE = lambda _: None


# bit reader
class BitReader():
    def __init__(self, data):
        self.data = data
        self.bpos = 0
        self.blen = 0
        self.bbuf = 0
    # read n-bits
    def bits(self, n):
        ret, s = 0, 0
        while 0 < n:
            if self.blen == 0:
                self.blen = 8
                self.bbuf = self.data[self.bpos]
                self.bpos += 1
            m = min(n, self.blen)
            mb = self.bbuf & ((1 << m) - 1)
            ret |= (mb << s)
            self.blen -= m
            self.bbuf >>= m
            n -= m
            s += m
        TRACE(f'BITS: {ret:0{s}b}')
        return ret
    # byte aligned?
    def byte_aligned(self):
        return self.blen == 0
assert BitReader(b'\x08\x02').bits(10) == 520


# Huffman decoder (RFC1951)
class HuffmanDecoder():
    def __init__(self, lens):
        # huffman lengths to huffman codes (RFC1951, 3.2.2)
        def len2code(lens):
            # step1
            MAX_BITS = max(lens)
            bl_count = [0] * (MAX_BITS + 1)
            for l in lens:
                bl_count[l] += 1
            # step2
            code = 0
            bl_count[0] = 0
            next_code = [0] * (MAX_BITS + 1)
            for bits in range(1, MAX_BITS + 1):
                code = (code + bl_count[bits-1]) << 1
                next_code[bits] = code
            # step3
            codes = [0] * len(lens)
            for n, l in enumerate(lens):
                if l != 0:
                    codes[n] = next_code[l]
                    next_code[l] += 1
            return codes
        self.lens = lens
        self.codes = len2code(lens)
        self.maxlen = max(lens)
        self.syms = {c: s for s, c in enumerate(self.codes) if lens[s] > 0}
    # decode symbol
    def decode(self, r):
        b, c = 1, r.bits(1)
        while b <= self.maxlen:
            if (b, c) in zip(self.lens, self.codes):
                TRACE(f'Huffman: {c:0{b}b} -> {self.syms[c]}')
                return self.syms[c]
            c = (c << 1) | r.bits(1)
            b += 1
        assert False, 'undefined huffman code'
    # debug: codes list
    def codes_str(self):
        return ', '.join([f'{c:0{l}b}' if l else '-' for c,l in zip(self.codes, self.lens)])
assert HuffmanDecoder([2,1,3,3]).codes == [0b10,0b0,0b110,0b111]
assert HuffmanDecoder([3,3,3,3,3,2,4,4]).codes == [0b010,0b011,0b100,0b101,0b110,0b00,0b1110,0b1111]


# Cyclic Redundancy Code (PNG Specifiction, Annex D)
def crc_lut(n):
    for _ in range(8):
        n = 0xedb88320 ^ (n >> 1) if n & 1 else n >> 1
    return n
CRC_TABLE = [crc_lut(n) for n in range(256)]


def calc_crc(data):
    def update_crc(c):
        for b in data:
            c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >> 8)
        return c
    return update_crc(0xffffffff) ^ 0xffffffff
assert calc_crc(b'IEND') == 0xae426082


# verify CRC of chunk
def verify_crc(f, chunk, data):
    crc = struct.unpack('>I', f.read(4))[0]
    print(f'  crc=0x{crc:08x}')
    assert crc == calc_crc(chunk[1] + data)


# parse image header(IHDR) chunk
def parse_IHDR(f, chunk):
    assert chunk == (13, b'IHDR')
    print(f'IHDR: length={chunk[0]}')
    data = f.read(13)
    K = ('width', 'height', 'bitdepth', 'color', 'compression', 'filter', 'interlace')
    ihdr = dict(zip(K, struct.unpack('>IIBBBBB', data)))
    for k in K:
        print(f'  {k}={ihdr[k]}')
    verify_crc(f, chunk, data)
    return ihdr


# parse image data(IDAT) chunk
def parse_IDAT(f, chunk):
    assert chunk[1] == b'IDAT'
    data = f.read(chunk[0])
    print(f'IDAT: length={chunk[0]}')
    verify_crc(f, chunk, data)
    return data


# parse image trailer(IEND) chunk
def parse_IEND(f, chunk):
    assert chunk == (0, b'IEND')
    print(f'IEND: length={chunk[0]}')
    verify_crc(f, chunk, b'')


# parse ZLIB stream (RFC1950)
def parse_zlib(data):
    assert len(data) > 6  # CMF(1)+FLG(1)+ADLER32(4)
    print(f'zlib stream: length={len(data)}')
    cmf, flg = data[0], data[1]
    cm, cinfo = cmf & 0xf, cmf >> 4
    fdict, flevel = (flg >> 5) & 1, flg >> 6
    print(f'  CM(Compression method)={cm}')
    print(f'  CINFO(Compression info)={cinfo} (window size={1<<(8+cinfo)})')
    print(f'  FDICT(Preset dictionary)={fdict}')
    print(f'  FLEVEL(Compression level)={flevel}')
    assert (cmf*256+flg) % 31 == 0, '{CMF,FLG} shall be multiple of 31'
    assert cm == 8, 'Support "deflate" method only'
    zlib_hdr = 2
    if dict == 1:
        dictid = data[2:6]
        print(f'  DICTID={dictid}')
        zlib_hdr += 4
    adler32 = struct.unpack('>I', data[len(data)-4:])[0]
    print(f'  ADLER32(Adler-32 checksum)=0x{adler32:08x}')
    return (data[zlib_hdr:len(data)-4], adler32)


# Adler-32 checksum (RFC1950, Appendix)
def calc_adler32(data):
    def update_adler32(adler, data):
        BASE = 65521
        s1, s2 = adler & 0xffff, (adler >> 16) & 0xffff
        for b in data:
            s1 = (s1 + b) % BASE
            s2 = (s2 + s1) % BASE
        return (s2 << 16) + s1
    return update_adler32(1, data)


DEFLATE_EXTRA_LENS = [
    (0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9), (0,10), # 257-264
    (1,11), (1,13), (1,15), (1,17),     # 265-268
    (2,19), (2,23), (2,27), (2,31),     # 269-272
    (3,35), (3,43), (3,51), (3,59),     # 273-276
    (4,67), (4,83), (4,99), (4,115),    # 277-280
    (5,131), (5,163), (5,195), (5,227), # 281-284
    (0,258)                             # 285
]
DEFLATE_EXTRA_DIST = [
    (0,1), (0,2), (0,3), (0,4), # 0-3
    (1,5), (1,7),           # 4-5
    (2,9), (2,13),          # 6-7
    (3,17), (3,25),         # 8-9
    (4,33), (4,49),         # 10-11
    (5,65), (5,97),         # 12-13
    (6,129), (6,193),       # 14-15
    (7,257), (7,385),       # 16-17
    (8,513), (8,769),       # 18-19
    (9,1025), (9,1537),     # 20-21
    (10,2049), (10,3073),   # 22-23
    (11,4097), (11,6145),   # 24-25
    (12,8193), (12,12289),  # 26-27
    (13,16385), (13,24577), # 28-29
]
DEFLATE_CLEN_ORD = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]


# decode DEFLATE/non-compressed block (RFC1951, 3.2.4)
def decode_deflate_raw(r, data):
    while not r.byte_aligned():
        r.bits(1)
    plen = r.bits(16)
    nlen = r.bits(16)
    print(f'    LEN={plen}')
    print(f'    NLEN={nlen}')
    assert plen ^ nlen == 0xffff, 'Invalid LEN/NLEN field'
    for _ in range(plen):
        data.append(r.bits(8))
    print(f'    (decode raw {plen} bytes)')


# decode DEFLATE/compressed block (RFC1951, 3.2.5)
def decode_deflate_data(r, hdec_lit, hdec_dist, data):
    print(f'    CompressedData')
    nsym, prevsize = 0, len(data)
    while True:
        value = hdec_lit.decode(r)
        nsym += 1
        assert 0 <= value <= 285, 'Invalid length code'
        if value < 256:
            # literal
            TRACE(f'      {value}')
            data.append(value)
        elif value == 256:
            # end-of-block
            TRACE(f'      EOB({value})')
            break
        else:
            # length and distance
            blen, bias = DEFLATE_EXTRA_LENS[value - 257]
            extra_bits = r.bits(blen)
            length = bias + extra_bits
            TRACE(f'      {value}: length={length} ({bias}+{extra_bits})')
            dist_code = hdec_dist.decode(r)
            blen, bias = DEFLATE_EXTRA_DIST[dist_code]
            extra_bits = r.bits(blen)
            dist = bias + extra_bits
            TRACE(f'      {value}: dist={dist} ({bias}+{extra_bits})')
            n = len(data) - dist
            for _ in range(length):
                data.append(data[n])
                n += 1
    print(f'    (decode {nsym} symbols to {len(data)-prevsize} bytes)')


# decode DEFLATE/fixed Huffman codes block (RFC1951, 3.2.6)
def decode_deflate_fixed(r, data):
    # 'literal and length alphabets' Huffman codes
    lit_lens = [8] * 144 + [9] * 112 + [7] * 24 + [8] * 8
    print(f'    LIT_LENS={lit_lens}')
    hdec_lit = HuffmanDecoder(lit_lens)
    print(f'    LIT_CODES=[{hdec_lit.codes_str()}]')
    # 'distance alphabets' Huffman codes
    dist_lens = [5] * 32
    print(f'    DIST_LENS={dist_lens}')
    hdec_dist = HuffmanDecoder(dist_lens)
    print(f'    DIST_CODES=[{hdec_dist.codes_str()}]')
    # decode compressed data
    return decode_deflate_data(r, hdec_lit, hdec_dist, data)


# decode DEFLATE/code lengths (RFC1951, 3.2.7)
def decode_deflate_codelens(r, hdec, size, sym):
    result = [0] * size
    n = 0
    while n < size:
        code = hdec.decode(r)
        assert 0 <= code <= 18, 'Invalid code length'
        if code <= 15:
            prev = result[n] = code
            TRACE(f'    {sym}[{n}]={code}')
            n += 1
        elif code == 16:
            extra_bits = r.bits(2)
            repeat = 3 + extra_bits
            TRACE(f'    {sym}[{n}..{n+repeat-1}]={prev} (code={code}/3+{extra_bits})')
            for _ in range(repeat):
                result[n] = prev
                n += 1
        else:
            if code == 17:
                blen, bias = (3, 3)  # 3-10 times
            else:
                blen, bias = (7, 11)  # 11-138 times
            extra_bits = r.bits(blen)
            repeat = bias + extra_bits
            prev = 0
            TRACE(f'    {sym}[{n}..{n+repeat-1}]={prev} (code={code}/{bias}+{extra_bits})')
            for _ in range(repeat):
                result[n] = 0
                n += 1
    return result


# decode DEFLATE/dynamic Huffman codes block (RFC1951, 3.2.7)
def decode_deflate_dynamic(r, data):
    hlit = r.bits(5)
    hdist = r.bits(5)
    hclen = r.bits(4)
    print(f'    HLIT={hlit} (literal/length codes={hlit + 257})')
    print(f'    HDIST={hdist} (distance codes={hdist + 1})')
    print(f'    HCLEN={hclen} (code length codes={hclen + 4})')
    # 'code length alphabet' Huffman codes
    clen = [r.bits(3) for _ in range(hclen + 4)]
    print(f'    CLEN={clen}')
    clen_lens = [0] * 19
    for n, o in zip_longest(clen, DEFLATE_CLEN_ORD, fillvalue=0):
        clen_lens[o] = n
    print(f'    CLEN_LENS={clen_lens}')
    hdec_clen = HuffmanDecoder(clen_lens)
    print(f'    CLEN_CODES=[{hdec_clen.codes_str()}]')
    # 'literal and length alphabets' Huffman codes
    lit_lens = decode_deflate_codelens(r, hdec_clen, hlit + 257, 'LIT')
    print(f'    LIT_LENS={lit_lens}')
    hdec_lit = HuffmanDecoder(lit_lens)
    print(f'    LIT_CODES=[{hdec_lit.codes_str()}]')
    # 'distance alphabets' Huffman codes
    dist_lens = decode_deflate_codelens(r, hdec_clen, hdist + 1, 'DIST')
    print(f'    DIST_LENS={dist_lens}')
    hdec_dist = HuffmanDecoder(dist_lens)
    print(f'    DIST_CODES=[{hdec_dist.codes_str()}]')
    # decode compressed data
    return decode_deflate_data(r, hdec_lit, hdec_dist, data)


# decode DEFLATE stream (RFC1951)
def decode_deflate(stream):
    print(f'deflate stream: length={len(stream)}')
    r = BitReader(stream)
    bfinal = 0
    data = bytearray()
    while bfinal == 0:
        bfinal = r.bits(1)
        btype = r.bits(2)
        print(f'  BTYPE={btype:b} BFINAL={bfinal}')
        assert btype != 0b11, 'BTYPE=0b11 is reserved'
        if btype == 0b00:
            # Non-compressed blocks
            decode_deflate_raw(r, data)
        elif btype == 0b01:
            # Compression with fixed Huffman codes
            decode_deflate_fixed(r, data)
        elif btype == 0b10:
            # Compression with dynamic Huffman codes
            decode_deflate_dynamic(r, data)
    print(f'(decode {r.bpos} bytes to {len(data)} bytes)')
    return data


# reconstruct image
def reconstruct_image(ihdr, data):
    width, height = ihdr['width'], ihdr['height']
    pixfmt = 'RGB' if ihdr['color'] == 2 else 'RGBA'
    print(f'image: {width}x{height}, {pixfmt}')
    pixsz = len(pixfmt)  # 3 or 4
    assert len(data) == (1 + width * pixsz) * height, 'Incrorect data size'
    pos, stride = 0, width * pixsz
    image = bytearray()
    prevline = bytearray([0] * stride)
    for y in range(height):
        ftype = data[pos]
        print(f'  line#{y}: filter={ftype}')
        assert 0 <= ftype <= 4, 'Invalid filter type'
        line = bytearray(data[pos+1:pos+1+stride])
        if ftype == 1:  # Sub
            for x in range(stride):
                a = line[x - pixsz] if x >= pixsz else 0
                line[x] = (line[x] + a) & 0xff
        elif ftype == 2:  # Up
            for x in range(stride):
                b = prevline[x]
                line[x] = (line[x] + b) & 0xff
        elif ftype == 3:  # Average
            for x in range(stride):
                a = line[x - pixsz] if x >= pixsz else 0
                b = prevline[x]
                line[x] = (line[x] + (a + b) // 2) & 0xff
        elif ftype == 4:  # Peath
            def pred(a, b, c):
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                if pa <= pb and pa <= pc:
                    return a
                if pb <= pc:
                    return b
                return c
            for x in range(stride):
                a = line[x - pixsz] if x >= pixsz else 0
                b = prevline[x]
                c = prevline[x - pixsz] if x >= pixsz else 0
                line[x] = (line[x] + pred(a, b, c)) & 0xff
        image += line
        prevline = line
        pos += 1 + stride
    return {'size': (width, height), 'data': image, 'pixfmt': pixfmt}


# parse PNG format
def parse_png(f):
    signature = f.read(8)
    print(f'Signature: {signature}')
    assert signature == b'\x89PNG\x0d\x0a\x1a\x0a', 'Invalid PNG signature'
    def get_chunk(f):
        return struct.unpack('>I4s', f.read(8))
    # first IHDR chunk
    ihdr = parse_IHDR(f, get_chunk(f))
    assert ihdr['bitdepth'] == 8 and ihdr['color'] in (2, 6), 'Support 8bit-Truecolour only'
    assert ihdr['interlace'] == 0, 'Support non-interlaced only'
    assert ihdr['filter'] == 0, 'Support filter method 0 only'
    # parse subsequent chunks
    idat = bytearray()
    while True:
        chunk = get_chunk(f)
        if chunk[1] == b'IDAT':
            idat += parse_IDAT(f, chunk)
        elif chunk[1] == b'IEND':
            parse_IEND(f, chunk)
            break  # last chunk
        else:
            # skip unrecognized chunk (include PLTE chunk)
            print(f'{chunk[1]}: length={chunk[0]}')
            data = f.read(chunk[0])
            verify_crc(f, chunk, data)
    # decode zlib/deflate stream
    stream, checksum = parse_zlib(idat)
    data = decode_deflate(stream)
    assert checksum == calc_adler32(data)
    # reconstruct image
    return reconstruct_image(ihdr, data)


# write image to PPM (portable pixmal format) file
def write_ppm(image, outfile):
    width, height = image['size']
    data = image['data']
    if image['pixfmt'] == 'RGBA':
        for n in range(width * height):
            if data[n*4+3] == 0:
                data[n*4:n*4+3] = (0, 0, 0)  # transparent -> black
        del data[3::4]  # remove alpha channel
    with open(outfile, 'wb') as f:
        f.write(f'P6\n{width} {height}\n255\n'.encode('ascii'))
        f.write(data)


def main(infile, outfile = None):
    with open(infile, 'rb') as f:
        image = parse_png(f)
        if outfile:
            write_ppm(image, outfile)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 1:
        print('usage: picopdec.py <input.png> [<output.ppm>]')
        sys.exit(1)
    main(*args)
