import enum
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import (BinaryIO, cast, Dict, Iterable,
                    List, NamedTuple, Optional, Set, Tuple)
import uuid
import zlib

from PIL import Image

from lxml import etree
from lxml.builder import ElementMaker


class ImageError(Exception):
    pass


NS = {'x': 'adobe:ns:meta/',
      'xmp': 'http://ns.adobe.com/xap/1.0/',
      'xmpRights': 'http://ns.adobe.com/xap/1.0/rights/',
      'xmpMM': 'http://ns.adobe.com/xap/1.0/mm/',
      'xmpidq': 'http://ns.adobe.com/xmp/Identifier/qual/1.0/',
      'stRef': 'http://ns.adobe.com/xap/1.0/sType/ResourceRef#',
      'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
      'dc': 'http://purl.org/dc/elements/1.1/'}


PNG_XMP_SIG = b'XML:com.adobe.xmp\x00\x00\x00\x00\x00'
PNG_TYPE = b'iTXt'
JPEG_XMP_SIG = b'http://ns.adobe.com/xap/1.0/\x00'


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
    # Sometimes there can be weird null bytes at the end
    data = raw_data.decode().rstrip('\x00')
    xml = etree.fromstring(data)
    if xml is None:
        return []
    yield from ((x.text or '').strip() for x
                in xml.xpath('//rdf:RDF/rdf:Description/dc:subject'
                             '/rdf:Bag/rdf:li', namespaces=NS)
                if (x.text or '').strip())


def create_xmp_chunk(tags: Set[str]) -> bytes:
    Ex = ElementMaker(namespace=NS['x'], nsmap=NS)
    Erdf = ElementMaker(namespace=NS['rdf'], nsmap=NS)
    Edc = ElementMaker(namespace=NS['dc'], nsmap=NS)
    tag_items = [Erdf.li(tag) for tag in sorted(tags)]
    root = Ex.xmpmeta(
        Erdf.RDF(
            Erdf.Description(
                Edc.subject(Erdf.Bag(*tag_items)),
                **{f'{{{NS["rdf"]}}}about': ''})))
    guid = uuid.uuid4().hex
    xpacket_start = f'<?xpacket begin="" id="{guid}"?>\n'
    xpacket_end = '\n<?xpacket end="w"?>'
    padding: str = 20 * ('\n' + ' ' * 99)
    out = ''.join([xpacket_start, etree.tostring(root, pretty_print=True,
                                                 encoding='unicode'),
                   padding, xpacket_end]).encode('utf-8')
    return out


def set_xmp_tags(raw_data: bytes, tags: Set[str]) -> bytes:
    x = '{' + NS['x'] + '}'
    rdf = '{' + NS['rdf'] + '}'
    dc = '{' + NS['dc'] + '}'
    about_key = rdf + 'about'
    data = raw_data.decode().rstrip('\x00')
    xml = etree.fromstring(data)
    paths = ['rdf:RDF', 'rdf:Description', 'dc:subject', 'rdf:Bag']
    xapmeta = 'xapmeta'
    xmpmeta = 'xmpmeta'
    if xml.tag == x + xapmeta:
        paths.insert(0, 'x:' + xapmeta)
    elif xml.tag == x + xmpmeta:
        paths.insert(0, 'x:' + xmpmeta)
    elif xml.tag != rdf + 'RDF':
        raise ImageError(f'Unrecognized root element: {xml.tag!r}')
    desc_abouts = set()
    for desc in xml.xpath('/' + '/'.join(paths[:-2]), namespaces=NS):
        about = desc.attrib.get(about_key, '')
        if about.strip():
            desc_abouts.add(about)
    parent = xml
    for i in range(len(paths)):
        hits = xml.xpath('/' + '/'.join(paths[:i+1]), namespaces=NS)
        target = paths[i]
        if len(hits) == 0:
            prefix, tag = target.split(':', 1)
            fullname = '{' + NS[prefix] + '}' + tag
            parent = etree.SubElement(parent, fullname)
            if target == 'rdf:Description' and len(desc_abouts) == 1:
                parent.set(about_key, list(desc_abouts)[0])
        else:
            # Sometimes there's one Description tag for each namespace,
            # and in that case we want to find the one with the dc namespace
            if target == 'rdf:Description':
                for hit in hits:
                    if NS['dc'] in hit.nsmap.values():
                        parent = hit
                        break
                else:
                    parent = hits[0]
            else:
                parent = hits[0]

    parent.clear()
    for tag in sorted(tags):
        tag_elem = etree.SubElement(parent, rdf + 'li')
        tag_elem.text = tag

    out_xml = etree.tostring(xml.getroottree(), pretty_print=True,
                             encoding='unicode').strip()
    if out_xml.endswith('?>'):
        start, end = out_xml.rsplit('<?', 1)
        end = '<?' + end
    max_len = 65503
    new_len = len(out_xml.encode())
    padding_len = len(raw_data) - new_len
    padding: str
    if padding_len > 0:
        padding = b''
        for n in range(0, padding_len, 100):
            padding += b'\n' + b' ' * 99
    else:
        padding = (20 * ('\n' + ' ' * 99)).encode()
        if len(padding) + new_len > max_len:
            padding_len = max_len - new_len

    out = start.encode() + padding[:padding_len] + end.encode()
    if len(out) > max_len:
        raise ImageError(f'New XMP length is too big ({new_len} > {max_len}, '
                         f'extended XMP not yet supported')
    return out


def png_dimensions(fname: Path) -> Tuple[int, int]:
    with open(fname, 'rb') as f:
        prefix = f.read(8)
        if prefix != b'\x89PNG\x0d\x0a\x1a\x0a':
            raise ImageError('not a png')
        # Skip IHDR length
        f.seek(4, 1)
        if f.read(4) != b'IHDR':
            raise ImageError('corrupted png')
        width, height = cast(Tuple[int, int],
                             struct.unpack('>II', f.read(8)))
        return width, height


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


def jpeg_dimensions(fname: Path) -> Tuple[int, int]:
    sof_markers = {b'\xc0', b'\xc1', b'\xc2', b'\xc3'}
    with open(fname, 'rb') as f:
        prefix = f.read(2)
        if prefix != b'\xff\xd8':
            raise ImageError('not a jpg')
        buf = b''
        while True:
            new_buf = f.read(1)
            # 0xD9 is the ending marker
            if len(new_buf) == 0 or buf + new_buf == b'\xff\xd9':
                break
            if new_buf == b'\xff':
                buf = new_buf
                continue
            if buf == b'\xff' and new_buf in sof_markers:
                # Skip size and bit depth
                f.seek(3, 1)
                height, width = cast(Tuple[int, int],
                                     struct.unpack('>HH', f.read(4)))
                return width, height
            if buf + new_buf == b'\xff\x00':
                buf = new_buf = b''
                continue
            maybe_length = f.read(2)
            if len(maybe_length) < 2:
                sys.stderr.write('Warning: premature end to image\n')
                break
            if maybe_length[0] == 0xff:
                f.seek(-len(maybe_length), 1)
                continue
            length = cast(Tuple[int], struct.unpack('>H', maybe_length))[0]
            if length:
                f.seek(length - 2, 1)
    return -1, -1


def parse_jpeg(f: BinaryIO) -> Tuple[Optional[int], Optional[int],
                                     Optional[Tuple[int, bytes]]]:
    prefix = f.read(2)
    if prefix != b'\xff\xd8':
        raise ImageError('not a jpg')
    buf = b''
    first_sof_pos: Optional[int] = None
    last_exif_pos: Optional[int] = None
    while True:
        new_buf = f.read(1)
        # 0xD9 is the ending marker
        if len(new_buf) == 0 or buf + new_buf == b'\xff\xd9':
            break
        if new_buf == b'\xff':
            buf = new_buf
            continue
        if buf == b'\xff' and new_buf in {b'\xc0', b'\xc2'} \
                and first_sof_pos is None:
            first_sof_pos = f.tell() - 2
        if buf + new_buf == b'\xff\x00':
            buf = new_buf = b''
            continue
        if buf + new_buf == b'\xff\xe1':
            start_pos = f.tell()
            length = cast(Tuple[int], struct.unpack('>H', f.read(2)))[0]
            data = f.read(length - 2)
            if len(data) != length - 2:
                sys.stderr.write('Warning: possibly corrupt jpg section\n')
            if data[:6] == b'Exif\x00\x00':
                # Find the last exif block
                last_exif_pos = f.tell()
            elif data[:len(JPEG_XMP_SIG)] == JPEG_XMP_SIG:
                # TODO: extended xmp?
                return first_sof_pos, last_exif_pos, (start_pos, data)
        else:
            maybe_length = f.read(2)
            if len(maybe_length) < 2:
                sys.stderr.write('Warning: premature end to image\n')
                break
            if maybe_length[0] == 0xff:
                f.seek(-len(maybe_length), 1)
                continue
            length = cast(Tuple[int], struct.unpack('>H', maybe_length))[0]
            if length:
                f.seek(length - 2, 1)
    return first_sof_pos, last_exif_pos, None


def read_jpeg_tags(fname: Path) -> Iterable[str]:
    with open(fname, 'rb') as f:
        _, _, payload = parse_jpeg(f)
        if payload is not None:
            yield from get_xmp_tags(payload[1][len(JPEG_XMP_SIG):])


def set_jpeg_tags(fname: Path, tags: Set[str]) -> None:
    with open(fname, 'r+b') as f:
        first_sof_pos, last_exif_pos, payload = parse_jpeg(f)
        if payload is not None:
            start_pos, data = payload
            new_data = JPEG_XMP_SIG + set_xmp_tags(data[len(JPEG_XMP_SIG):],
                                                   tags)
            if len(new_data) != len(data):
                f.seek(start_pos + len(data) + 2)
                trailing_data = f.read()
                f.seek(start_pos)
                f.truncate()
                f.write(struct.pack('>H', len(new_data) + 2))
                f.write(new_data)
                f.write(trailing_data)
            elif new_data != data:
                f.seek(start_pos + 2)
                f.write(new_data)
            return
        else:
            if last_exif_pos is not None:
                start_pos = last_exif_pos
            elif first_sof_pos is not None:
                start_pos = first_sof_pos
            else:
                raise ImageError('Weird jpeg, can\'t find place '
                                 'to put XMP block')
            f.seek(start_pos)
            trailing_data = f.read()
            f.seek(start_pos)
            new_data = JPEG_XMP_SIG + create_xmp_chunk(tags)
            f.write(b'\xff\xe1')
            f.write(struct.pack('>H', len(new_data) + 2))
            f.write(new_data)
            f.write(trailing_data)


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


def read_tags(fname: Path) -> Set[str]:
    formats = {
        ImageFormat.GIF: read_gif_tags,
        ImageFormat.JPEG: read_jpeg_tags,
        ImageFormat.PNG: read_png_tags,
    }
    try:
        fmt = identify_image_format(fname)
    except ImageError as e:
        return set()
    else:
        if fmt is not None:
            return set(formats[fmt](fname))
        else:
            return set()


def set_tags(fname: Path, tags: Set[str], safe: bool = True) -> None:
    # - get xmp/iptc/exif data from exiv2
    # - load image into memory
    # - copy file to tmp
    # - run set_*_tags on copy
    # - get xmp/iptc/exif/image data and compare
    # - copy file back to original path
    def get_metadata(path: Path) -> List[str]:
        try:
            result = subprocess.run(['exiv2', '-p', 'a', str(path)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    check=True, encoding='utf-8')
        except subprocess.CalledProcessError as e:
            raise ImageError(f'running exiv2 failed!\nstdout: {e.stdout!r}'
                             f'\nstderr: {e.stderr!r}')
        else:
            return [x for x in result.stdout.splitlines()
                    if not x.startswith('Xmp.dc.subject ')]

    def get_pixeldata(path: Path) -> List[Tuple[int, ...]]:
        return list(Image.open(str(path)).getdata())

    set_tags_funcs = {
        ImageFormat.PNG: set_png_tags,
        ImageFormat.JPEG: set_jpeg_tags,
    }
    fmt = identify_image_format(fname)
    if fmt is None:
        raise ImageError('unknown file format')

    if not safe:
        set_tags_funcs[fmt](fname, tags)
        return

    old_meta = get_metadata(fname)
    old_image = get_pixeldata(fname)

    with tempfile.TemporaryDirectory(prefix='jfti-inprogress-') as tempdir:
        # Copy file to safe temporary place
        new_file = shutil.copy(str(fname), tempdir)
        # Set the tags
        set_tags_funcs[fmt](new_file, tags)
        # Get the data and check it
        new_meta = get_metadata(new_file)
        try:
            new_image = get_pixeldata(new_file)
        except OSError:
            raise ImageError('couldnt read the new file!')
        if new_meta != old_meta:
            if sorted(new_meta) == sorted(old_meta):
                raise ImageError('metadata out of order!')
            new_lines = set(new_meta) - set(old_meta)
            dropped_lines = set(old_meta) - set(new_meta)
            raise ImageError(f'metadata do not match!\n'
                             f'added lines: {new_lines!r}\n'
                             f'removed lines: {dropped_lines!r}')
        if new_image != old_image:
            print(list(new_image)[0])
            print(type(list(new_image)[0]))
            print(len(new_image))
            print(len(old_image))
            raise ImageError('pixel data do not match!')
        shutil.copy(new_file, str(fname))


def dimensions(fname: Path) -> Tuple[int, int]:
    formats = {
        ImageFormat.JPEG: jpeg_dimensions,
        ImageFormat.PNG: png_dimensions,
    }
    try:
        fmt = identify_image_format(fname)
    except ImageError:
        return -1, -1
    else:
        if fmt is not None:
            return formats[fmt](fname)
        else:
            return -1, -1


def print_tags(fname: Path) -> None:
    formats = {
        ImageFormat.GIF: read_gif_tags,
        ImageFormat.JPEG: read_jpeg_tags,
        ImageFormat.PNG: read_png_tags,
    }
    try:
        fmt = identify_image_format(fname)
    except ImageError as e:
        sys.exit(str(e))
    if fmt is None:
        sys.exit('Unsupported file format')
    print(', '.join(list(formats[fmt](fname))))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('action', choices=('set', 'ls'))
    parser.add_argument('tag', nargs='*')
    parser.add_argument('--unsafe', action='store_true')
    args = parser.parse_args()
    path = Path(args.file).expanduser()
    if args.action == 'ls':
        if args.tag:
            raise argparse.ArgumentError('dont specify tags when listing')
        print_tags(path)
    elif args.action == 'set':
        if not args.tag:
            raise argparse.ArgumentError('no tags specified (no you cant '
                                         'get rid of all of them yet)')
        set_tags(path, set(args.tag), safe=not args.unsafe)


if __name__ == '__main__':
    main()
