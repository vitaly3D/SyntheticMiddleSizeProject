#!/usr/bin/env python3
"""
Generate independent, content-unique clones of a Unity asset folder.

Intended for import-time / project-open / Accelerator benchmarking: every clone
gets fresh GUIDs, all references inside the clone are remapped to the clone's
own assets, and every binary source file is modified so that no two copies
share the same bytes (no dedupe in VCS, no hash collisions in caches).

Per-type handling:
  .meta / JSON / text          GUID references remapped
  YAML assets                  GUID references remapped + "# CloneNN" comment appended
  binary Unity assets          GUID references remapped (nibble-swapped 16-byte form)
  .png                         one channel of the last pixel shifted (palette images: last palette entry)
  .fbx (binary + ASCII)        first vertex (or first animation key) nudged by ~1e-5 relative
  .exr                         extra header attribute (offset table fixed up)
  .tif/.tiff                   extra ASCII tag in a rewritten IFD
  .tga                         last pixel byte shifted
  .pcache (binary)             last float nudged
  .shader / .shadergraph       shader name / graph path made unique per clone
  .hlsl/.cginc                 comment appended
Scripts (.cs, .asmdef, ...) are NOT cloned; clones keep referencing the originals.

Pure Python, no third-party dependencies.

Usage:
  python clone_scenes.py --src Assets/Scenes --dst Assets/Clones --count 10
  python clone_scenes.py --src Assets/Scenes --dst Assets/Clones --start 3 --count 1
  python clone_scenes.py --self-test Assets/Scenes   # round-trip check of the binary handlers
"""
import argparse
import hashlib
import os
import re
import shutil
import struct
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

EXCLUDED_EXT = {'.cs', '.asmdef', '.asmref', '.dll', '.so', '.dylib', '.rsp', '.jslib', '.aar', '.jar'}
UNITY_SERIALIZED_EXT = {
    '.unity', '.prefab', '.mat', '.asset', '.anim', '.controller', '.overridecontroller', '.playable',
    '.signal', '.lighting', '.giparams', '.preset', '.mask', '.physicmaterial', '.physicsmaterial2d',
    '.cubemap', '.flare', '.rendertexture', '.mixer', '.guiskin', '.fontsettings', '.spriteatlas',
    '.graphicsstate', '.terrainlayer', '.brush', '.vfx', '.vfxoperator', '.vfxblock',
}
GUID_TEXT_RE = re.compile(rb'(?<![0-9a-fA-F])[0-9a-f]{32}(?![0-9a-fA-F])')
META_GUID_RE = re.compile(r'^guid: ([0-9a-f]{32})', re.M)

FOLDER_META = (
    'fileFormatVersion: 2\nguid: {guid}\nfolderAsset: yes\nDefaultImporter:\n'
    '  externalObjects: {{}}\n  userData: \n  assetBundleName: \n  assetBundleVariant: \n'
)


class Unsupported(Exception):
    pass


def clone_tag(k):
    return f'Clone{k:02d}'


def new_guid(old, k):
    return hashlib.md5(f'{old}:{clone_tag(k)}'.encode()).hexdigest()


def nibble_swap(guid_hex):
    return bytes(((x & 15) << 4) | (x >> 4) for x in bytes.fromhex(guid_hex))


# ---------------------------------------------------------------- GUID remap

def remap_text(data, gmap):
    return GUID_TEXT_RE.sub(lambda m: gmap.get(m.group(0), m.group(0)), data)


def remap_binary(data, bmap):
    out = data
    for old, new in bmap.items():
        if old in out:
            out = out.replace(old, new)
    return out


# ---------------------------------------------------------------- PNG

PNG_SIG = b'\x89PNG\r\n\x1a\n'


def png_chunk(typ, payload):
    return struct.pack('>I', len(payload)) + typ + payload + struct.pack('>I', zlib.crc32(typ + payload) & 0xffffffff)


def mutate_png(data, k):
    if data[:8] != PNG_SIG:
        raise Unsupported('not a PNG')
    pos, chunks = 8, []
    while pos < len(data):
        ln, = struct.unpack_from('>I', data, pos)
        typ = data[pos + 4:pos + 8]
        chunks.append((typ, data[pos + 8:pos + 8 + ln]))
        pos += 12 + ln
        if typ == b'IEND':
            break
    w, h, bd, ct, _, _, il = struct.unpack('>IIBBBBB', chunks[0][1])
    if ct == 3:
        # Palette image: shift the blue channel of the last palette entry.
        out = bytearray(PNG_SIG)
        for typ, payload in chunks:
            if typ == b'PLTE':
                payload = bytearray(payload)
                payload[-1] = (payload[-1] + k) & 255
                payload = bytes(payload)
            out += png_chunk(typ, payload)
        return bytes(out) + data[pos:]
    if il:
        raise Unsupported('interlaced PNG')
    idat_idx = [i for i, (t, _) in enumerate(chunks) if t == b'IDAT']
    raw = bytearray(zlib.decompress(b''.join(chunks[i][1] for i in idat_idx)))
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[ct]
    stride = (w * channels * bd + 7) // 8
    if len(raw) != h * (stride + 1):
        raise Unsupported('unexpected IDAT size')
    # The very last byte of the stream is the last sample of the last pixel of the last row.
    # No other byte is predicted from it, so only that one sample changes (by exactly k).
    raw[-1] = (raw[-1] + k) & 255
    comp = zlib.compress(bytes(raw), 9)
    out = bytearray(PNG_SIG)
    first, last = idat_idx[0], idat_idx[-1]
    for i, (typ, payload) in enumerate(chunks):
        if i == first:
            for off in range(0, len(comp), 1 << 16):
                out += png_chunk(b'IDAT', comp[off:off + (1 << 16)])
        elif first < i <= last:
            continue
        else:
            out += png_chunk(typ, payload)
    return bytes(out) + data[pos:]  # keep any bytes trailing IEND


# ---------------------------------------------------------------- TGA

def mutate_tga(data, k):
    idlen, cmtype, imtype = data[0], data[1], data[2]
    if imtype not in (1, 2, 3):
        raise Unsupported('compressed TGA')
    cmlen, cmbits = struct.unpack_from('<H', data, 5)[0], data[7]
    w, h, bpp = struct.unpack_from('<HHB', data, 12)
    start = 18 + idlen + (cmlen * ((cmbits + 7) // 8) if cmtype else 0)
    end = start + w * h * ((bpp + 7) // 8)
    if end > len(data):
        raise Unsupported('truncated TGA')
    out = bytearray(data)
    out[end - 1] = (out[end - 1] + k) & 255
    return bytes(out)


# ---------------------------------------------------------------- TIFF

def mutate_tiff(data, k):
    e = {b'II': '<', b'MM': '>'}.get(data[:2])
    if e is None or struct.unpack_from(e + 'H', data, 2)[0] != 42:
        raise Unsupported('not a classic TIFF')
    ifd, = struct.unpack_from(e + 'I', data, 4)
    n, = struct.unpack_from(e + 'H', data, ifd)
    entries = [data[ifd + 2 + 12 * i: ifd + 14 + 12 * i] for i in range(n)]
    next_ifd = data[ifd + 2 + 12 * n: ifd + 6 + 12 * n]
    tags = {struct.unpack_from(e + 'H', x)[0] for x in entries}
    tag = 270 if 270 not in tags else 65000  # ImageDescription, else a private tag
    if tag in tags:
        raise Unsupported('tag already present')
    text = f'SyntheticMiddleSizeProject {clone_tag(k)}\0'.encode()
    out = bytearray(data)
    if len(out) % 2:
        out += b'\0'
    text_off = len(out)
    out += text
    if len(out) % 2:
        out += b'\0'
    new_ifd = len(out)
    entries.append(struct.pack(e + 'HHII', tag, 2, len(text), text_off))
    entries.sort(key=lambda x: struct.unpack_from(e + 'H', x)[0])
    out += struct.pack(e + 'H', len(entries)) + b''.join(entries) + next_ifd
    struct.pack_into(e + 'I', out, 4, new_ifd)
    return bytes(out)


# ---------------------------------------------------------------- EXR

EXR_LINES_PER_BLOCK = {0: 1, 1: 1, 2: 1, 3: 16, 4: 32, 5: 16, 6: 32, 7: 32, 8: 32, 9: 256}


def mutate_exr(data, k):
    if data[:4] != b'\x76\x2f\x31\x01':
        raise Unsupported('not an EXR')
    flags = struct.unpack_from('<I', data, 4)[0] >> 8
    if flags & (0x2 | 0x8 | 0x10):  # tiled, deep, multipart
        raise Unsupported('tiled/deep/multipart EXR')
    pos, comp, dw = 8, None, None
    while data[pos] != 0:
        name_end = data.index(0, pos)
        name = data[pos:name_end]
        type_end = data.index(0, name_end + 1)
        size, = struct.unpack_from('<I', data, type_end + 1)
        val = type_end + 5
        if name == b'compression':
            comp = data[val]
        elif name == b'dataWindow':
            dw = struct.unpack_from('<iiii', data, val)
        pos = val + size
    header_end = pos + 1
    if comp not in EXR_LINES_PER_BLOCK or dw is None:
        raise Unsupported('unknown EXR compression')
    lpb = EXR_LINES_PER_BLOCK[comp]
    count = (dw[3] - dw[1] + 1 + lpb - 1) // lpb
    offsets = struct.unpack_from(f'<{count}Q', data, header_end)
    if min(offsets) != header_end + 8 * count:
        raise Unsupported('unexpected EXR offset table')
    value = f'{clone_tag(k)}'.encode()
    attr = b'syntheticClone\0string\0' + struct.pack('<I', len(value)) + value
    shift = len(attr)
    return (data[:pos] + attr + b'\0'
            + struct.pack(f'<{count}Q', *(o + shift for o in offsets))
            + data[header_end + 8 * count:])


# ---------------------------------------------------------------- FBX

FBX_MAGIC = b'Kaydara FBX Binary  \x00\x1a\x00'


class FbxNode:
    __slots__ = ('name', 'nprops', 'props', 'children', 'nested')

    def __init__(self, name, nprops, props, children, nested):
        self.name, self.nprops, self.props, self.children, self.nested = name, nprops, props, children, nested


def _fbx_read(b, pos, wide):
    if wide:
        end, nprops, plen = struct.unpack_from('<QQQ', b, pos)
        pos += 24
    else:
        end, nprops, plen = struct.unpack_from('<III', b, pos)
        pos += 12
    nlen = b[pos]
    pos += 1
    if end == 0:
        return None, pos
    name = b[pos:pos + nlen]
    pos += nlen
    props = b[pos:pos + plen]
    pos += plen
    children, nested = [], pos < end
    while pos < end:
        child, pos = _fbx_read(b, pos, wide)
        if child is None:
            break
        children.append(child)
    if pos != end:
        raise Unsupported('FBX node size mismatch')
    return FbxNode(name, nprops, props, children, nested), pos


def _fbx_write(out, node, wide):
    hdr = 25 if wide else 13
    start = len(out)
    out += b'\0' * hdr
    out += node.name + node.props
    for c in node.children:
        _fbx_write(out, c, wide)
    if node.nested:
        out += b'\0' * hdr
    fmt = '<QQQB' if wide else '<IIIB'
    struct.pack_into(fmt, out, start, len(out), node.nprops, len(node.props), len(node.name))


def fbx_parse(data):
    if not data.startswith(FBX_MAGIC):
        raise Unsupported('not a binary FBX')
    version, = struct.unpack_from('<I', data, 23)
    wide = version >= 7500
    pos, nodes = 27, []
    while True:
        node, pos = _fbx_read(data, pos, wide)
        if node is None:
            break
        nodes.append(node)
    return version, wide, nodes, pos, data[pos:]


def fbx_serialize(version, wide, nodes, orig_end, footer):
    out = bytearray(FBX_MAGIC + struct.pack('<I', version))
    for n in nodes:
        _fbx_write(out, n, wide)
    out += b'\0' * (25 if wide else 13)
    # Footer: 16-byte id, 4 zero bytes, alignment padding, then 140 fixed bytes.
    # Keep the writer's alignment convention by preserving the original (offset + pad) residue.
    pad = len(footer) - 20 - 140
    if pad < 0:
        raise Unsupported('unexpected FBX footer')
    residue = (orig_end + 20 + pad) % 16
    base = len(out) + 20
    new_pad = (residue - base) % 16
    if pad >= 16 or (new_pad == 0 and pad == 16):
        new_pad += 16 * (pad // 16)
    out += footer[:20] + b'\0' * new_pad + footer[20 + pad:]
    return bytes(out)


def _fbx_props(raw):
    props, pos = [], 0
    while pos < len(raw):
        t = raw[pos:pos + 1]
        start = pos
        pos += 1
        if t in b'YCIFDL':
            pos += {b'Y': 2, b'C': 1, b'I': 4, b'F': 4, b'D': 8, b'L': 8}[t]
        elif t in b'fdlib':
            _, _, clen = struct.unpack_from('<III', raw, pos)
            pos += 12 + clen
        elif t in b'SR':
            ln, = struct.unpack_from('<I', raw, pos)
            pos += 4 + ln
        else:
            raise Unsupported(f'unknown FBX property type {t!r}')
        props.append(raw[start:pos])
    return props


def _nudge(v, k):
    return v + max(abs(v), 1.0) * 1e-5 * k


def _nudge_array_prop(p, k):
    t = p[:1]
    if t not in b'fd':
        return None
    n, enc, clen = struct.unpack_from('<III', p, 1)
    if n == 0:
        return None
    payload = p[13:13 + clen]
    arr = bytearray(zlib.decompress(payload) if enc == 1 else payload)
    fmt = '<d' if t == b'd' else '<f'
    struct.pack_into(fmt, arr, 0, _nudge(struct.unpack_from(fmt, arr, 0)[0], k))
    payload = zlib.compress(bytes(arr)) if enc == 1 else bytes(arr)
    return t + struct.pack('<III', n, enc, len(payload)) + payload


def _fbx_find(nodes, name):
    for n in nodes:
        if n.name == name:
            yield n
        yield from _fbx_find(n.children, name)


def mutate_fbx_binary(data, k):
    version, wide, nodes, end, footer = fbx_parse(data)
    for target in (b'Vertices', b'KeyValueFloat'):
        for node in _fbx_find(nodes, target):
            props = _fbx_props(node.props)
            if props:
                new = _nudge_array_prop(props[0], k)
                if new is not None:
                    node.props = new + b''.join(props[1:])
                    return fbx_serialize(version, wide, nodes, end, footer)
    raise Unsupported('no vertex or key data in FBX')


FBX_ASCII_NUM_RE = re.compile(
    rb'((?:Vertices|KeyValueFloat):\s*(?:\*\d+\s*\{\s*a:\s*)?)([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)')


def mutate_fbx_ascii(data, k):
    m = FBX_ASCII_NUM_RE.search(data)
    if not m:
        return data + f'\n; {clone_tag(k)}\n'.encode()
    v = _nudge(float(m.group(2)), k)
    return data[:m.start(2)] + repr(v).encode() + data[m.end(2):]


def mutate_fbx(data, k):
    if data.startswith(FBX_MAGIC):
        return mutate_fbx_binary(data, k)
    return mutate_fbx_ascii(data, k)


# ---------------------------------------------------------------- misc text/binary

def mutate_pcache(data, k):
    hdr_end = data.find(b'end_header')
    if hdr_end < 0:
        raise Unsupported('bad pcache')
    if b'format binary' in data[:hdr_end]:
        out = bytearray(data)
        v, = struct.unpack_from('<f', out, len(out) - 4)
        struct.pack_into('<f', out, len(out) - 4, _nudge(v, k))
        return bytes(out)
    return data.rstrip() + f' {k * 1e-6}\n'.encode()


SHADER_NAME_RE = re.compile(rb'(\bShader\s*")([^"]+)(")')
SHADERGRAPH_PATH_RE = re.compile(rb'("m_Path"\s*:\s*")([^"]*)(")')


def mutate_shader(data, k):
    return SHADER_NAME_RE.sub(lambda m: m.group(1) + m.group(2) + f'_{clone_tag(k)}'.encode() + m.group(3), data, 1)


def mutate_shadergraph(data, k):
    tag = clone_tag(k).encode()
    return SHADERGRAPH_PATH_RE.sub(
        lambda m: m.group(1) + (m.group(2) + b'/' + tag if m.group(2) else tag) + m.group(3), data, 1)


def mutate_code_include(data, k):
    return data + f'\n// {clone_tag(k)}\n'.encode()


BINARY_HANDLERS = {
    '.png': mutate_png, '.tga': mutate_tga, '.tif': mutate_tiff, '.tiff': mutate_tiff,
    '.exr': mutate_exr, '.fbx': mutate_fbx, '.pcache': mutate_pcache,
}
TEXT_HANDLERS = {
    '.shader': mutate_shader, '.shadergraph': mutate_shadergraph, '.shadersubgraph': mutate_shadergraph,
    '.hlsl': mutate_code_include, '.cginc': mutate_code_include,
}


def is_text(data):
    return b'\0' not in data[:8192]


# ---------------------------------------------------------------- worker

_GMAPS = {}
_BMAPS = {}


def _init_worker(gmaps):
    _GMAPS.update(gmaps)
    for k, gm in gmaps.items():
        _BMAPS[k] = {nibble_swap(o.decode()): nibble_swap(n.decode()) for o, n in gm.items()}


def process_file(src, dst, k):
    """Returns (handler_name, warning or None)."""
    ext = os.path.splitext(src)[1].lower()
    gmap = _GMAPS[k]
    with open(src, 'rb') as f:
        data = f.read()
    warning = None
    if ext == '.meta':
        out, kind = remap_text(data, gmap), 'meta'
    elif ext in BINARY_HANDLERS:
        kind = ext
        try:
            out = BINARY_HANDLERS[ext](data, k)
        except Unsupported as e:
            out, warning = data, f'{src}: copied unchanged ({e})'
    elif is_text(data):
        out, kind = remap_text(data, gmap), 'text'
        if ext in TEXT_HANDLERS:
            out, kind = TEXT_HANDLERS[ext](out, k), ext
        elif out.startswith(b'%YAML'):
            # Assets without clone-internal references would otherwise be identical across clones.
            out, kind = out.rstrip(b'\r\n') + f'\n# {clone_tag(k)}\n'.encode(), 'yaml'
    elif ext in UNITY_SERIALIZED_EXT:
        out, kind = remap_binary(data, _BMAPS[k]), 'unity-binary'
    else:
        out, kind = data, 'copied'
        warning = f'{src}: unknown binary type, copied unchanged'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, 'wb') as f:
        f.write(out)
    if out == data and ext != '.meta' and kind not in ('text', 'unity-binary', 'copied'):
        warning = warning or f'{src}: content unchanged'
    return kind, warning


# ---------------------------------------------------------------- planning

def plan(src_root):
    """Collect files to clone (relative paths, metas included) and the GUIDs they own."""
    files, guids = [], []
    keep_dirs = set()
    for dirpath, dirnames, filenames in os.walk(src_root):
        dirnames.sort()
        for fn in sorted(filenames):
            if fn.endswith('.meta'):
                continue
            if os.path.splitext(fn)[1].lower() in EXCLUDED_EXT:
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), src_root)
            files.append(rel)
            d = os.path.dirname(rel)
            while d:
                keep_dirs.add(d)
                d = os.path.dirname(d)
    entries = files + sorted(keep_dirs)
    out = []
    for rel in entries:
        meta = rel + '.meta'
        meta_path = os.path.join(src_root, meta)
        if not os.path.exists(meta_path):
            raise SystemExit(f'missing meta file: {meta_path}')
        with open(meta_path, encoding='utf-8', errors='replace') as f:
            m = META_GUID_RE.search(f.read())
        if not m:
            raise SystemExit(f'no guid in {meta_path}')
        guids.append(m.group(1))
        out.append(meta)
    return files + out, guids


def write_folder_meta(path, guid):
    with open(path + '.meta', 'w', newline='\n') as f:
        f.write(FOLDER_META.format(guid=guid))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default='Assets/Scenes')
    ap.add_argument('--dst', default='Assets/Clones')
    ap.add_argument('--start', type=int, default=1, help='first clone index')
    ap.add_argument('--count', type=int, default=10, help='number of clones')
    ap.add_argument('--jobs', type=int, default=os.cpu_count())
    ap.add_argument('--force', action='store_true', help='replace clone folders that already exist')
    ap.add_argument('--self-test', metavar='DIR', help='round-trip test the binary handlers on DIR and exit')
    args = ap.parse_args()

    if args.self_test:
        return self_test(args.self_test)

    if not 1 <= args.start or args.start + args.count - 1 > 255:
        raise SystemExit('clone indices must be within 1..255')
    clones = list(range(args.start, args.start + args.count))
    rel_files, guids = plan(args.src)
    print(f'{len(rel_files)} files ({len(guids)} assets) per clone, clones {clones[0]}..{clones[-1]}')

    gmaps = {k: {g.encode(): new_guid(g, k).encode() for g in guids} for k in clones}

    os.makedirs(args.dst, exist_ok=True)
    if not os.path.exists(args.dst.rstrip('/\\') + '.meta'):
        write_folder_meta(args.dst.rstrip('/\\'), hashlib.md5(b'SyntheticClonesRoot').hexdigest())
    for k in clones:
        root = os.path.join(args.dst, clone_tag(k))
        if os.path.exists(root):
            if not args.force:
                raise SystemExit(f'{root} exists (use --force to replace)')
            shutil.rmtree(root)
        os.makedirs(root)
        write_folder_meta(root, hashlib.md5(f'SyntheticClonesRoot:{clone_tag(k)}'.encode()).hexdigest())

    sizes = {rel: os.path.getsize(os.path.join(args.src, rel)) for rel in rel_files}
    tasks = sorted(((rel, k) for k in clones for rel in rel_files), key=lambda t: -sizes[t[0]])
    t0, done, stats, warnings = time.time(), 0, {}, []
    with ProcessPoolExecutor(args.jobs, initializer=_init_worker, initargs=(gmaps,)) as ex:
        futs = [ex.submit(process_file, os.path.join(args.src, rel),
                          os.path.join(args.dst, clone_tag(k), rel), k) for rel, k in tasks]
        for fut in as_completed(futs):
            kind, warning = fut.result()
            stats[kind] = stats.get(kind, 0) + 1
            if warning:
                warnings.append(warning)
            done += 1
            if done % 500 == 0 or done == len(futs):
                print(f'  {done}/{len(futs)} files, {time.time() - t0:.0f}s', flush=True)

    total = sum(os.path.getsize(os.path.join(dp, f)) for k in clones
                for dp, _, fs in os.walk(os.path.join(args.dst, clone_tag(k))) for f in fs)
    print('handled:', ', '.join(f'{k}={v}' for k, v in sorted(stats.items())))
    print(f'written: {total / 2**30:.2f} GiB in {time.time() - t0:.0f}s')
    for w in sorted(set(warnings)):
        print('WARNING', w)


# ---------------------------------------------------------------- self test

def self_test(root):
    """Verify the binary handlers produce parseable, different output on every file under root."""
    failures = 0
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in BINARY_HANDLERS:
                continue
            path = os.path.join(dirpath, fn)
            data = open(path, 'rb').read()
            try:
                if ext == '.fbx' and data.startswith(FBX_MAGIC):
                    parsed = fbx_parse(data)
                    if fbx_serialize(*parsed) != data:
                        raise AssertionError('FBX identity round-trip differs')
                out = BINARY_HANDLERS[ext](data, 1)
                if out == data:
                    raise AssertionError('unchanged')
                if ext == '.png':
                    _png_check(out, data)
                elif ext == '.fbx' and out.startswith(FBX_MAGIC):
                    fbx_parse(out)
                elif ext == '.exr':
                    _exr_check(out, data)
            except Exception as e:  # noqa: BLE001 - report everything
                failures += 1
                print(f'FAIL {path}: {type(e).__name__}: {e}')
    print('self-test', 'FAILED' if failures else 'passed', f'({failures} failures)')
    return 1 if failures else 0


def _png_idat(data):
    pos, idat, crc_ok = 8, b'', True
    while pos < len(data):
        ln, = struct.unpack_from('>I', data, pos)
        typ, payload = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + ln]
        crc_ok &= struct.unpack_from('>I', data, pos + 8 + ln)[0] == zlib.crc32(typ + payload) & 0xffffffff
        if typ == b'IDAT':
            idat += payload
        pos += 12 + ln
        if typ == b'IEND':
            break
    assert crc_ok, 'bad CRC'
    return zlib.decompress(idat)


def _png_check(out, orig):
    a, b = _png_idat(out), _png_idat(orig)
    assert len(a) == len(b)
    diff = sum(1 for x, y in zip(a[-64:], b[-64:]) if x != y) + (a[:-64] != b[:-64]) * 1000
    assert diff <= 1, f'{diff} bytes differ'


def _exr_check(out, orig):
    # Every chunk referenced by the new offset table must match the original chunk.
    def chunks(d):
        pos = 8
        while d[pos] != 0:
            pos = d.index(0, d.index(0, pos) + 1) + 1
            pos += 4 + struct.unpack_from('<I', d, pos)[0]
        pos += 1
        first, = struct.unpack_from('<Q', d, pos)
        n = (first - pos) // 8
        return [d[o:o + 8 + struct.unpack_from('<i', d, o + 4)[0]] for o in struct.unpack_from(f'<{n}Q', d, pos)]
    assert chunks(out) == chunks(orig), 'EXR chunks differ'


if __name__ == '__main__':
    sys.exit(main())
