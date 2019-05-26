import enum
from pathlib import Path
import re
import struct
import sys
from typing import (BinaryIO, cast, Dict, Iterable,
                    List, NamedTuple, Optional, Set, Tuple)
import uuid
import xml.etree.ElementTree as ET
import zlib


class ImageError(Exception):
    pass


NS = {'x': 'adobe:ns:meta/',
      'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
      'dc': 'http://purl.org/dc/elements/1.1/'}

for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


PNG_XMP_SIG = b'XML:com.adobe.xmp\x00\x00\x00\x00\x00'
PNG_TYPE = b'iTXt'


class ImageFormat(enum.Enum):
    GIF = enum.auto()
    JPEG = enum.auto()
    PNG = enum.auto()


def identify_image_format(fname: Path) -> Optional[ImageFormat]:
    with open(fname, 'rb') as f:
        data = f.read(8)
    if data == b'\x89PNG\x0d\x0a\x1a\x0a':
        return ImageFormat.PNG
    elif data[:2] == b'\xff\xd8':
        return ImageFormat.JPEG
    elif data[:3] == b'GIF':
        if data[3:6] == b'89a':
            return ImageFormat.GIF
        else:
            raise ImageError(f'Unsupported GIF format: {data[3:6]}')
    return None


def get_xmp_tags(raw_data: bytes) -> Iterable[str]:
    data = raw_data.decode()
    xml = ET.fromstring(data)
    # TODO: handle missing xmpmeta
    keyword_bag = xml.find('rdf:RDF/rdf:Description'
                           '/dc:subject/rdf:Bag', NS)
    for keyword in keyword_bag or []:
        if keyword.text:
            yield keyword.text


def create_xmp_chunk(tags: Set[str]) -> bytes:
    root = ET.Element('x:xmpmeta',
                      attrib={'xmlns:' + k: v for k, v in NS.items()})
    paths: List[Tuple[str, Dict[str, str]]] = [
        ('rdf:RDF', {}),
        ('rdf:Description', {'rdf:about': ''}),
        ('dc:subject', {}),
        ('rdf:Bag', {})
    ]
    parent = root
    indent = '\n'
    for tag, attribs in paths:
        parent.text = indent + ' '
        parent = ET.SubElement(parent, tag, attrib=attribs)
        parent.tail = indent
        indent += ' '
    parent.text = indent + ' '
    for n, tag in enumerate(sorted(tags)):
        tag_elem = ET.SubElement(parent, 'rdf:li')
        tag_elem.text = tag
        tag_elem.tail = indent + (' ' if n < len(tags) - 1 else '')
    guid = uuid.uuid4().hex
    xpacket_start = f'<?xpacket begin="" id="{guid}"?>\n'
    xpacket_end = '\n<?xpacket end="w"?>'
    padding: str = 20 * ('\n' + ' ' * 99)
    out = ''.join([xpacket_start, ET.tostring(root, encoding='unicode'),
                   padding, xpacket_end]).encode('utf-8')
    return out


def set_xmp_tags(raw_data: bytes, tags: Set[str]) -> bytes:
    data = raw_data.decode()
    xml = ET.fromstring(data)
    paths: List[Tuple[str, Dict[str, str]]] = [
        ('rdf:RDF', {}),
        ('rdf:Description', {'rdf:about': ''}),
        ('dc:subject', {}),
        ('rdf:Bag', {})
    ]
    xpacket_start = re.match(r'<\?xpacket [^>]+\?>', data)
    xpacket_end = re.search(r'<\?xpacket end=["\'][rw]["\']\?>$', data)
    assert xpacket_start is not None and xpacket_end is not None
    parent = xml
    for tag, attribs in paths:
        child = parent.find(tag, NS)
        if child is None:
            parent = ET.SubElement(parent, tag, attrib=attribs)
        else:
            parent = child
    for keyword in list(parent):
        parent.remove(keyword)
    indent = '\n ' + ' ' * len(paths)
    for tag in sorted(tags):
        tag_elem = ET.SubElement(parent, 'rdf:li')
        tag_elem.text = tag
        tag_elem.tail = indent
    if len(parent) > 0:
        parent[-1].tail = indent[:-1]
    out_xml = ET.tostring(xml, encoding='unicode')
    out = '\n'.join([xpacket_start[0], out_xml]).encode()
    end = b'\n' + xpacket_end[0].encode()
    padding_len = max(0, len(raw_data) - (len(out) + len(end)))
    padding = b''
    for n in range(0, padding_len, 100):
        padding += b'\n' + b' ' * 99
    out += padding[:padding_len] + end
    return out


class PNGChunk(NamedTuple):
    pos: int
    data: bytes


def parse_png(f: BinaryIO) -> Tuple[int, List[PNGChunk]]:
    prefix = f.read(8)
    if prefix != b'\x89PNG\x0d\x0a\x1a\x0a':
        raise ImageError('not a png')
    block_pos: int
    chunks = []
    first_pos: int = -1
    while True:
        block_pos = f.tell()
        length_bytes = f.read(4)
        if len(length_bytes) == 0:
            break
        length = cast(Tuple[int], struct.unpack('>I', length_bytes))[0]
        type_ = f.read(4)
        if type_ == b'IEND':
            break
        data = f.read(length)
        crc = cast(Tuple[int], struct.unpack('>I', f.read(4)))[0]
        assert crc == zlib.crc32(type_ + data)
        if type_ == PNG_TYPE and data[:len(PNG_XMP_SIG)] == PNG_XMP_SIG:
            chunks.append(PNGChunk(block_pos, data))
        if first_pos < 0:
            first_pos = f.tell()
    return first_pos, chunks


def read_png_tags(fname: Path) -> Iterable[str]:
    with open(fname, 'rb') as f:
        _, chunks = parse_png(f)
        for pos, data in chunks:
            yield from get_xmp_tags(data[len(PNG_XMP_SIG):])


def set_png_tags(fname: Path, tags: Set[str]) -> None:
    with open(fname, 'r+b') as f:
        start_pos, chunks = parse_png(f)
        if len(chunks) == 1:
            pos, data = chunks[0]
            new_data = PNG_XMP_SIG + set_xmp_tags(data[len(PNG_XMP_SIG):],
                                                  tags)
            if len(new_data) != len(data):
                new_crc = zlib.crc32(PNG_TYPE + new_data)
                f.seek(pos + 8 + len(data) + 4)
                trailing_data = f.read()
                f.seek(pos)
                f.write(struct.pack('>I', len(new_data)))
                f.write(PNG_TYPE + new_data)
                f.write(struct.pack('>I', new_crc))
                f.write(trailing_data)
            elif new_data != data:
                new_crc = zlib.crc32(PNG_TYPE + new_data)
                f.seek(pos + 8)
                f.write(new_data)
                f.write(struct.pack('>I', new_crc))
        elif not chunks:
            f.seek(start_pos)
            trailing_data = f.read()
            f.seek(start_pos)
            xml = PNG_XMP_SIG + create_xmp_chunk(tags)
            f.write(struct.pack('>I', len(xml)))
            f.write(PNG_TYPE + xml)
            crc = zlib.crc32(PNG_TYPE + xml)
            f.write(struct.pack('>I', crc))
            f.write(trailing_data)


def read_jpeg_tags(fname: Path) -> Iterable[str]:
    with open(fname, 'rb') as f:
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
                length = cast(Tuple[int], struct.unpack('>H', f.read(2)))[0]
                data = f.read(length - 2)
                if data[:sig_len] == start_signature:
                    yield from get_xmp_tags(data[sig_len:])
            else:
                maybe_length = f.read(2)
                if maybe_length[0] == 0xff:
                    f.seek(-2, 1)
                    continue
                length = cast(Tuple[int], struct.unpack('>H', maybe_length))[0]
                if length:
                    f.seek(length - 2, 1)


def read_gif_tags(fname: Path) -> Iterable[str]:
    def skip_color_table(f: BinaryIO, packed_data: int) -> None:
        has_color_table = packed_data >> 7
        if has_color_table:
            size = packed_data & 0b111
            f.seek(3 * cast(int, 2 ** (size + 1)), 1)

    def skip_sub_blocks(f: BinaryIO) -> None:
        while True:
            size = f.read(1)[0]
            if size == 0:
                break
            f.seek(size, 1)

    with open(fname, 'rb') as f:
        prefix = f.read(6)
        if prefix == b'GIF87a':
            raise ImageError('gif version 87a not supported')
        elif prefix[:3] == b'GIF' and prefix[3:] != b'89a':
            raise ImageError(f'unknown gif version: {prefix[3:]}')
        elif prefix != b'GIF89a':
            raise ImageError('not a gif')
        # skip last two bytes in logical screen descriptor
        w, h, packed = cast(Tuple[int, int, int],
                            struct.unpack('<HHBxx', f.read(7)))
        skip_color_table(f, packed)
        TEXT = 0x01
        GCE = 0xf9
        COMMENT = 0xfe
        APP = 0xff
        while True:
            label = f.read(2)
            if len(label) == 0:
                break
            # Extension block
            if label[0] == 0x21:
                if label[1] in {TEXT, GCE, COMMENT}:
                    # Skip irrelevant parts
                    skip_sub_blocks(f)
                elif label[1] == APP:
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
                else:
                    print('UNK ext', hex(label[1]))
            elif label[0] == 0x2c:
                f.seek(-1, 1)
                x, y, w, h, packed = cast(Tuple[int, int, int, int, int],
                                          struct.unpack('<HHHHB', f.read(9)))
                skip_color_table(f, packed)
                # Skip LZW min code size
                f.seek(1, 1)
                skip_sub_blocks(f)
            elif label[0] == 0x3b:
                shit = f.read(10)
                if len(shit) > 0:
                    print('trailing data:', shit)
                break
            else:
                print('UNK', hex(label[0]))


def print_tags(fname: Path) -> None:
    formats = {
        # ImageFormat.GIF: read_gif_tags,
        # ImageFormat.JPEG: read_jpeg_tags,
        # ImageFormat.PNG: read_png_tags,
        ImageFormat.PNG: set_png_tags,
    }
    try:
        fmt = identify_image_format(fname)
    except ImageError as e:
        sys.exit(str(e))
    if fmt is None:
        sys.exit('Unsupported file format')
    formats[fmt](fname, set(['nah', 'lolololo']))
    # print(', '.join(list(formats[fmt](fname))))


if __name__ == '__main__':
    print_tags(Path(sys.argv[1]).expanduser())
