from pathlib import Path
import struct
import sys
from typing import Iterable, BinaryIO
import xml.etree.ElementTree as ET


class ImageError(Exception):
    pass


def get_xmp_tags(data: bytes) -> Iterable[str]:
    ns = {'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
          'dc': 'http://purl.org/dc/elements/1.1/'}
    xml = ET.fromstring(data.decode())
    keyword_bag = xml.find('rdf:RDF/rdf:Description'
                           '/dc:subject/rdf:Bag', ns)
    for keyword in keyword_bag or []:
        if keyword.text:
            yield keyword.text


def read_png_tags(fname: Path) -> Iterable[str]:
    chunks = []
    with fname.open('rb') as f:
        prefix = f.read(8)
        if prefix != b'\x89PNG\x0d\x0a\x1a\x0a':
            raise ImageError('not a png')
        for _ in range(200):
            length_bytes = f.read(4)
            if len(length_bytes) == 0:
                break
            length = struct.unpack('>I', length_bytes)[0]
            type_ = f.read(4)
            if type_ == b'IEND':
                break
            data = f.read(length)
            crc = f.read(4)
            chunks.append((type_, length, crc, data))

    start_signature = b'XML:com.adobe.xmp\x00\x00\x00\x00\x00'
    sig_len = len(start_signature)
    for type_, length, crc, data in chunks:
        if type_ != b'iTXt':
            continue
        if data[:sig_len] == start_signature:
            yield from get_xmp_tags(data[sig_len:])


def read_jpeg_tags(fname: Path) -> Iterable[str]:
    with fname.open('rb') as f:
        prefix = f.read(2)
        if prefix != b'\xff\xd8':
            raise ImageError('not a jpg')
        buf = b''
        start_signature = b'http://ns.adobe.com/xap/1.0/\x00'
        sig_len = len(start_signature)
        while True:
            new_buf = f.read(1)
            if len(new_buf) == 0:
                break
            if new_buf == b'\xff':
                buf = new_buf
                continue
            if buf == b'\xff' and new_buf == b'\x00':
                buf = new_buf = b''
                continue
            if buf + new_buf == b'\xff\xe1':
                length = struct.unpack('>H', f.read(2))[0]
                data = f.read(length - 2)
                if data[:sig_len] == start_signature:
                    yield from get_xmp_tags(data[sig_len:])
            else:
                maybe_length = f.read(2)
                if maybe_length[0] == 0xff:
                    f.seek(-2, 1)
                    continue
                length = struct.unpack('>H', maybe_length)[0]
                if length:
                    f.seek(length - 2, 1)
    return []


def read_gif_tags(fname: Path) -> Iterable[str]:
    def skip_color_table(f: BinaryIO, packed_data: int) -> None:
        has_color_table = packed >> 7
        if has_color_table:
            size = packed & 0b111
            f.seek(3 * 2 ** (size + 1), 1)

    with fname.open('rb') as f:
        prefix = f.read(6)
        if prefix == b'GIF87a':
            raise ImageError('gif version 87a not supported')
        elif prefix[:3] == b'GIF' and prefix[3:] != b'89a':
            raise ImageError(f'unknown gif version: {prefix[3:]}')
        elif prefix != b'GIF89a':
            raise ImageError('not a gif')
        w, h, packed = struct.unpack('<HHB', f.read(5))
        # skip last two bytes in logical screen descriptor
        f.seek(2, 1)
        skip_color_table(f, packed)
        while True:
            label = f.read(2)
            if len(label) == 0:
                break
            # Extension block
            if label[0] == 0x21:
                if label[1] == 0xf9:
                    # Graphic control extension
                    size = f.read(1)[0]
                    assert size == 4
                    f.seek(size, 1)
                    assert f.read(1)[0] == 0
                elif label[1] == 0xfe:
                    # Comment extension
                    comment = b''
                    while True:
                        size = f.read(1)[0]
                        if size == 0:
                            break
                        comment += f.read(size)
                elif label[1] == 0xff:
                    # Application extension
                    assert f.read(1)[0] == 11
                    app_id = f.read(8)
                    app_auth = f.read(3)
                    data = b''
                    is_xmp = (app_id == b'XMP Data' and app_auth == b'XMP')
                    while True:
                        size = f.read(1)[0]
                        if size == 0:
                            break
                        if is_xmp:
                            f.seek(-1, 1)
                            data += f.read(size + 1)
                        else:
                            data += f.read(size)
                    if is_xmp:
                        yield from get_xmp_tags(data[:-257])
                elif label[1] == 0x01:
                    # Plain text extension
                    assert f.read(1)[0] == 12
                    f.seek(12, 1)
                    while True:
                        size = f.read(1)[0]
                        if size == 0:
                            break
                        f.seek(size, 1)
                else:
                    print('UNK ext', hex(label[1]))
            elif label[0] == 0x2c:
                f.seek(-1, 1)
                x, y, w, h, packed = struct.unpack('<HHHHB', f.read(9))
                skip_color_table(f, packed)
                # Skip LZW min code size
                f.seek(1, 1)
                while True:
                    size = f.read(1)[0]
                    if size == 0:
                        break
                    f.seek(size, 1)
            elif label[0] == 0x3b:
                shit = f.read(10)
                if len(shit) > 0:
                    print('trailing data:', shit)
                break
            else:
                print('UNK', hex(label[0]))
    return []


def print_tags(fname: Path) -> None:
    # TODO: check for magic data instead
    if fname.suffix.lower() == '.png':
        print(', '.join(list(read_png_tags(fname))))
    elif fname.suffix.lower() == '.jpg':
        print(', '.join(list(read_jpeg_tags(fname))))
    elif fname.suffix.lower() == '.gif':
        print(', '.join(list(read_gif_tags(fname))))
    else:
        sys.exit('unsupported file format')


if __name__ == '__main__':
    print_tags(Path(sys.argv[1]).expanduser())