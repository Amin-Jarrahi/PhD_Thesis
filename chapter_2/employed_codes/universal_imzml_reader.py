# universal_imzml_reader.py
import xml.etree.ElementTree as ET
import struct
import numpy as np
import zlib
import base64
from typing import Dict, Any, Tuple, List, Optional, Generator

# ---------------------------
# MSNumpress (linear/pic/slof) decoders
# (kept as your original implementations, slightly refactored)
# ---------------------------
def msnumpress_linear_decode(byte_arr: bytes, dtype=np.float64) -> np.ndarray:
    import math
    if len(byte_arr) == 0:
        return np.array([], dtype=dtype)

    pos = 0
    # try header count
    count = None
    if len(byte_arr) >= 4:
        maybe = struct.unpack("<I", byte_arr[0:4])[0]
        if 0 <= maybe <= 10_000_000 and maybe * 8 <= (len(byte_arr) + 100):
            pos = 4
            count = maybe

    out = []
    b = byte_arr
    n = len(b)
    i = pos

    def read_varint():
        nonlocal i
        if i >= n:
            raise EOFError
        result = 0
        shift = 0
        while True:
            byte = b[i]
            i += 1
            result |= (byte & 0x7F) << shift
            if (byte & 0x80) == 0:
                break
            shift += 7
        # zigzag
        val = (result >> 1) ^ (-(result & 1))
        return val

    # try to read scale (double) if available
    scale = None
    if i + 8 <= n:
        try:
            scale = struct.unpack("<d", b[i:i+8])[0]
            i += 8
        except struct.error:
            scale = None
    if scale is None:
        scale = 1.0

    prev = 0.0
    try:
        while i < n:
            vi = read_varint()
            val = prev + (vi / scale)
            out.append(val)
            prev = val
            if count is not None and len(out) >= count:
                break
    except EOFError:
        pass

    return np.array(out, dtype=dtype)


def numpress_decode_pic(byte_arr: bytes, dtype=np.float64) -> np.ndarray:
    if len(byte_arr) == 0:
        return np.array([], dtype=dtype)

    pos = 0
    count = None
    if len(byte_arr) >= 4:
        maybe = struct.unpack("<I", byte_arr[0:4])[0]
        if 0 <= maybe <= 10_000_000 and maybe * 8 <= (len(byte_arr) + 100):
            pos = 4
            count = maybe

    def read_varint():
        nonlocal pos
        result = 0
        shift = 0
        while True:
            if pos >= len(byte_arr):
                raise IndexError
            byte = byte_arr[pos]
            pos += 1
            result |= (byte & 0x7F) << shift
            if (byte & 0x80) == 0:
                break
            shift += 7
        return result

    out = []
    prev = 0
    try:
        while pos < len(byte_arr):
            v = read_varint()
            prev += v
            out.append(prev)
            if count is not None and len(out) >= count:
                break
    except IndexError:
        pass
    return np.array(out, dtype=dtype)


def numpress_decode_slof(byte_arr: bytes, dtype=np.float64) -> np.ndarray:
    import math
    if len(byte_arr) == 0:
        return np.array([], dtype=dtype)

    pos = 0
    count = None
    if len(byte_arr) >= 4:
        maybe = struct.unpack("<I", byte_arr[0:4])[0]
        if 0 <= maybe <= 10_000_000 and maybe * 8 <= (len(byte_arr) + 100):
            pos = 4
            count = maybe

    def read_varint():
        nonlocal pos
        result = 0
        shift = 0
        while True:
            if pos >= len(byte_arr):
                raise IndexError
            byte = byte_arr[pos]
            pos += 1
            result |= (byte & 0x7F) << shift
            if (byte & 0x80) == 0:
                break
            shift += 7
        return result

    out = []
    prev = 0
    try:
        while pos < len(byte_arr):
            v = read_varint()
            prev += v
            decoded = math.exp(prev / 100000.0) - 1.0
            out.append(decoded)
            if count is not None and len(out) >= count:
                break
    except IndexError:
        pass
    return np.array(out, dtype=dtype)


# ---------------------------
# utility helpers
# ---------------------------
def _get_cv_params(elem) -> Dict[str, str]:
    """Return dict accession -> value for cvParam children of elem."""
    out = {}
    if elem is None:
        return out
    for cv in elem.findall(".//mzml:cvParam", {"mzml": "http://psi.hupo.org/ms/mzml"}):
        acc = cv.attrib.get("accession")
        # cvParam may or may not have value attribute
        val = cv.attrib.get("value")
        if acc:
            out[acc] = val
    return out


def _bytes_to_array(raw_bytes: bytes, dtype_char: str, byteorder: str) -> np.ndarray:
    """Convert raw bytes to numpy array with dtype char mapping (f/d/i/q)."""
    if raw_bytes is None or len(raw_bytes) == 0:
        return np.array([], dtype=np.float32 if dtype_char == "f" else np.float64)

    char_to_np = {"f": ("<f4", 4), "d": ("<f8", 8), "i": ("<i4", 4), "q": ("<i8", 8)}
    if dtype_char not in char_to_np:
        raise ValueError(f"Unsupported dtype char: {dtype_char}")

    np_dtype_le, size = char_to_np[dtype_char]
    if byteorder == ">":
        np_dtype = np_dtype_le.replace("<", ">")
    else:
        np_dtype = np_dtype_le

    count = len(raw_bytes) // size
    return np.frombuffer(raw_bytes[: count * size], dtype=np.dtype(np_dtype)).copy()


# ---------------------------
# Universal Reader
# ---------------------------
class UniversalImzMLReader:
    """
    Universal imzML + IBD reader.
    Usage:
        r = UniversalImzMLReader("file.imzML", "file.ibd")
        mzs, ints = r.getspectrum(0)
        mzs, ints, x, y = r.getspectrum_with_coords(0)
    """

    def __init__(self, imzml_path: str, ibd_path: Optional[str] = None):
        self.imzml_path = imzml_path
        self.ibd_path = ibd_path
        self.ns = {"mzml": "http://psi.hupo.org/ms/mzml"}
        self.spectra_meta: List[Dict[str, Any]] = []
        self._ref_param_groups: Dict[str, Dict[str, str]] = {}
        self._parse_imzml()

    def _parse_imzml(self):
        tree = ET.parse(self.imzml_path)
        root = tree.getroot()

        # parse referenceableParamGroupList -> map id -> cv params
        rpg_list = root.find(".//mzml:referenceableParamGroupList", self.ns)
        if rpg_list is not None:
            for pg in rpg_list.findall("mzml:referenceableParamGroup", self.ns):
                pid = pg.attrib.get("id")
                if pid:
                    self._ref_param_groups[pid] = _get_cv_params(pg)

        # iterate spectra
        for i, spectrum in enumerate(root.findall(".//mzml:spectrum", self.ns)):
            # coordinates (common IMS accessions)
            x = None
            y = None
            scan = spectrum.find(".//mzml:scan", self.ns)
            if scan is not None:
                x_cv = scan.find("./mzml:cvParam[@accession='IMS:1000050']", self.ns)  # x
                y_cv = scan.find("./mzml:cvParam[@accession='IMS:1000051']", self.ns)  # y
                if x_cv is not None and x_cv.attrib.get("value") is not None:
                    try:
                        x = int(float(x_cv.attrib["value"]))
                    except Exception:
                        x = None
                if y_cv is not None and y_cv.attrib.get("value") is not None:
                    try:
                        y = int(float(y_cv.attrib["value"]))
                    except Exception:
                        y = None

            # find binaryDataArray elements (usually first is m/z, second is intensity)
            bdal = spectrum.findall(".//mzml:binaryDataArray", self.ns)
            if not bdal:
                # fallback: empty
                self.spectra_meta.append({
                    "index": i, "x": x, "y": y, "arrays": []
                })
                continue

            arrays = []
            for bda in bdal:
                # gather CV params from this BDA
                local_cv = _get_cv_params(bda)

                # also include cv params from any referenceableParamGroupRef children (if present)
                for ref in bda.findall("mzml:referenceableParamGroupRef", self.ns):
                    rid = ref.attrib.get("ref")
                    if rid and rid in self._ref_param_groups:
                        # merge (local wins)
                        for k, v in self._ref_param_groups[rid].items():
                            if k not in local_cv:
                                local_cv[k] = v

                # check for cv params in parent binaryDataArrayList or in spectrum-level refs?
                # (some files store compression/precision in param groups referenced at spectrum)
                for ref in spectrum.findall("mzml:referenceableParamGroupRef", self.ns):
                    rid = ref.attrib.get("ref")
                    if rid and rid in self._ref_param_groups:
                        for k, v in self._ref_param_groups[rid].items():
                            if k not in local_cv:
                                local_cv[k] = v

                # detect compression (allow multiple accessions present)
                compression = None
                # Check both zlib + numpress combinations
                has_zlib = "MS:1000574" in local_cv
                has_linear = "MS:1002312" in local_cv
                has_pic = "MS:1002313" in local_cv
                has_slof = "MS:1002314" in local_cv

                if has_zlib and has_linear:
                    compression = "numpress_linear+zlib"
                elif has_zlib and has_pic:
                    compression = "numpress_pic+zlib"
                elif has_zlib and has_slof:
                    compression = "numpress_slof+zlib"
                elif has_linear:
                    compression = "numpress_linear"
                elif has_pic:
                    compression = "numpress_pic"
                elif has_slof:
                    compression = "numpress_slof"
                elif has_zlib:
                    compression = "zlib"
                else:
                    compression = None

                # precision
                precision = None
                if "MS:1000521" in local_cv:  # 32-bit float
                    precision = "f"
                elif "MS:1000523" in local_cv:  # 64-bit float
                    precision = "d"
                elif "MS:1000519" in local_cv:  # 32-bit int
                    precision = "i"
                elif "MS:1000522" in local_cv:  # 64-bit int
                    precision = "q"
                else:
                    # default to 32-bit float if not declared (common)
                    precision = "f"

                # byteorder
                byteorder = "<"  # little-endian default
                if "MS:1000140" in local_cv:
                    byteorder = ">"
                elif "MS:1000141" in local_cv:
                    byteorder = "<"

                # inline binary (base64) if present
                raw_inline = None
                binary_elem = bda.find("mzml:binary", self.ns)
                if binary_elem is not None and binary_elem.text and binary_elem.text.strip():
                    # base64 decode
                    try:
                        raw_inline = base64.b64decode(binary_elem.text.strip())
                    except Exception:
                        raw_inline = None

                # offsets/lengths: both IMS accessions and explicit tags possible
                offset = None
                length = None
                # often stored as cvParam with IMS accessions
                if "IMS:1000102" in local_cv:
                    try:
                        offset = int(local_cv["IMS:1000102"])
                    except Exception:
                        offset = None
                if "IMS:1000104" in local_cv:
                    try:
                        length = int(local_cv["IMS:1000104"])
                    except Exception:
                        length = None

                # some imzML use <offset> and <encodedLength> child tags (different writers)
                off_tag = bda.find("mzml:offset", self.ns)
                if off_tag is not None and off_tag.text:
                    try:
                        offset = int(off_tag.text)
                    except Exception:
                        pass
                len_tag = bda.find("mzml:encodedLength", self.ns)
                if len_tag is not None and len_tag.text:
                    try:
                        length = int(len_tag.text)
                    except Exception:
                        pass

                arrays.append({
                    "compression": compression,
                    "precision": precision,
                    "byteorder": byteorder,
                    "offset": offset,
                    "length": length,
                    "inline": raw_inline  # raw bytes if inline
                })

            self.spectra_meta.append({
                "index": i,
                "x": x,
                "y": y,
                "arrays": arrays
            })

    # ---------- low-level binary read ----------
    def _read_raw_block(self, arr_meta: Dict[str, Any]) -> bytes:
        """Return raw bytes for a binaryDataArray element: either inline or read from IBD."""
        if arr_meta.get("inline") is not None:
            return arr_meta["inline"] or b""
        offset = arr_meta.get("offset")
        length = arr_meta.get("length")
        if offset is None or length is None or length <= 0:
            return b""
        if not self.ibd_path:
            raise ValueError("No IBD path provided but spectrum references external data.")
        with open(self.ibd_path, "rb") as f:
            f.seek(offset)
            return f.read(length)

    def _decode_block(self, raw_bytes: bytes, meta: Dict[str, Any]) -> np.ndarray:
        """Decode raw_bytes according to meta (compression, precision, byteorder)."""
        compression = meta.get("compression")
        precision = meta.get("precision") or "f"
        byteorder = meta.get("byteorder") or "<"

        # helper: raw typed array
        def typed_from(bts):
            return _bytes_to_array(bts, precision, ">" if byteorder == ">" else "<")

        if raw_bytes is None or len(raw_bytes) == 0:
            return np.array([])

        # handle combined/composed compressions
        # If compression contains both numpress and zlib, decompress then numpress decode
        if compression is None:
            return typed_from(raw_bytes)

        # zlib-only
        if compression == "zlib":
            dec = zlib.decompress(raw_bytes)
            return typed_from(dec)

        # numpress linear (maybe compressed by zlib)
        if compression in ("numpress_linear", "numpress_linear+zlib"):
            bts = raw_bytes
            if compression.endswith("+zlib"):
                bts = zlib.decompress(raw_bytes)
            return msnumpress_linear_decode(bts, dtype=np.float64)

        # numpress pic
        if compression in ("numpress_pic", "numpress_pic+zlib"):
            bts = raw_bytes
            if compression.endswith("+zlib"):
                bts = zlib.decompress(raw_bytes)
            return numpress_decode_pic(bts, dtype=np.float64)

        # numpress slof
        if compression in ("numpress_slof", "numpress_slof+zlib"):
            bts = raw_bytes
            if compression.endswith("+zlib"):
                bts = zlib.decompress(raw_bytes)
            return numpress_decode_slof(bts, dtype=np.float64)

        raise NotImplementedError(f"Compression '{compression}' not supported by this reader.")

    # ---------- public API ----------
    def getspectrum(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (m/z array, intensity array) like pyimzML's getspectrum.
        Assumes arrays[0] is m/z and arrays[1] is intensity (common case).
        """
        if index < 0 or index >= len(self.spectra_meta):
            raise IndexError("spectrum index out of range")
        meta = self.spectra_meta[index]
        arrays = meta.get("arrays", [])
        if len(arrays) < 2:
            return np.array([]), np.array([])

        # read both blocks
        mz_meta = arrays[0]
        int_meta = arrays[1]
        raw_mz = self._read_raw_block(mz_meta)
        raw_int = self._read_raw_block(int_meta)

        mz_vals = self._decode_block(raw_mz, mz_meta)
        int_vals = self._decode_block(raw_int, int_meta)

        # align lengths
        if mz_vals.size != int_vals.size:
            minlen = min(mz_vals.size, int_vals.size)
            mz_vals = mz_vals[:minlen]
            int_vals = int_vals[:minlen]

        # filter non-finite
        valid = np.isfinite(mz_vals) & np.isfinite(int_vals)
        if np.any(~valid):
            mz_vals = mz_vals[valid]
            int_vals = int_vals[valid]

        return mz_vals, int_vals

    def getspectrum_with_coords(self, index: int) -> Tuple[np.ndarray, np.ndarray, Optional[int], Optional[int]]:
        """Return (mz, intensity, x, y) if coordinates are present."""
        mz, it = self.getspectrum(index)
        md = self.spectra_meta[index]
        return mz, it, md.get("x"), md.get("y")

    def unique_mzs(self) -> np.ndarray:
        all_mzs = []
        for i in range(len(self.spectra_meta)):
            mz, _ = self.getspectrum(i)
            if mz.size > 0:
                all_mzs.append(mz)
        if not all_mzs:
            return np.array([])
        return np.unique(np.concatenate(all_mzs))

    def sparse_matrix_generator(self, batch_size: int = 500) -> Generator[Tuple[List[int], List[int], List[float], np.ndarray], None, None]:
        n = len(self.spectra_meta)
        for start in range(0, n, batch_size):
            rows, cols, values = [], [], []
            coords = []
            mzs_in_batch = []
            for i in range(start, min(start + batch_size, n)):
                mzv, iv = self.getspectrum(i)
                coords.append((self.spectra_meta[i]["x"], self.spectra_meta[i]["y"]))
                mzs_in_batch.append(mzv)
            if len(mzs_in_batch) == 0:
                yield [], [], [], np.array(coords)
                continue
            all_mzs = np.unique(np.concatenate([arr for arr in mzs_in_batch if arr.size > 0]))
            mz_to_index = {mz: j for j, mz in enumerate(all_mzs)}
            for local_row, i in enumerate(range(start, min(start + batch_size, n))):
                mzv, iv = self.getspectrum(i)
                if mzv.size == 0:
                    continue
                valid = iv != 0
                mzv = mzv[valid]
                iv = iv[valid]
                for m, inten in zip(mzv, iv):
                    rows.append(local_row)
                    cols.append(mz_to_index[m])
                    values.append(float(inten))
            yield rows, cols, values, np.array(coords)


# Convenience factory
def open_imzml_ibd(imzml_path: str, ibd_path: Optional[str] = None) -> UniversalImzMLReader:
    return UniversalImzMLReader(imzml_path, ibd_path)
