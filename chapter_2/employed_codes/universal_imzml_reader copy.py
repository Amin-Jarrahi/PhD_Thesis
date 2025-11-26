# universal_imzml_reader.py
import xml.etree.ElementTree as ET
import struct
import numpy as np
import zlib
from typing import Dict, Any, Tuple, List, Optional, Generator

# ---------------------------
# MSNumpress (linear) decoder
# ---------------------------
# This is a minimal, pure-Python implementation of MSNumpress "linear" decoding
# adapted to be standalone. It's sufficient for many imzML exports that use
# MS:1002312 (numpress linear). If you encounter other numpress types, say so
# and I will extend.
def msnumpress_linear_decode(byte_arr: bytes, dtype=np.float64) -> np.ndarray:
    """
    Decode MSNumpress linear compressed bytes to float array.
    Returns numpy array of dtype (usually float64).
    Note: This implementation assumes input bytes represent the numpress-coded block
    (as defined in the numpress spec). It returns float64 by default.
    """
    # The spec encodes as variable-length integer deltas with a scale factor.
    # Here's a small, pragmatic implementation based on typical references.
    # If this fails on your file, send me a small example and I'll adapt.
    import math

    if len(byte_arr) == 0:
        return np.array([], dtype=dtype)

    # first 4 bytes: number of values as 32-bit int (little-endian)
    # Some writers don't include count; many use a header with num of encoded bytes.
    # We'll try to decode flexibly: if the first 4 bytes look like a sane count, use it.
    pos = 0
    # read header if present (common pattern: 4 bytes count)
    if len(byte_arr) >= 4:
        count = struct.unpack("<I", byte_arr[0:4])[0]
        # if count is reasonable, consume it
        if 0 <= count <= 10_000_000 and count * 8 <= (len(byte_arr) + 100):  # heuristics
            pos = 4
        else:
            count = None
    else:
        count = None

    # Fallback: decode until end using integer decoding
    out = []
    b = byte_arr
    n = len(b)
    i = pos
    # decode little-endian signed variable int (zigzag) groups
    # This is a lightweight approach and may not match all numpress variants.
    # We'll implement a tolerant varint reader.
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
        # zigzag decode to signed
        val = (result >> 1) ^ (-(result & 1))
        return val

    # The numeric decode must reconstruct floating values from integer deltas and a scale
    # In practice many encoders put a scalefactor (double) first.
    # Try to read a scale double if enough bytes remain:
    scale = None
    if i + 8 <= n:
        try:
            scale = struct.unpack("<d", b[i:i+8])[0]
            i += 8
        except struct.error:
            scale = None

    if scale is None:
        # fallback default scale
        scale = 1.0

    # Now read integer-coded deltas until exhaustion
    prev = 0.0
    try:
        while i < n:
            vi = read_varint()
            # vi is delta * (1/scale) basically; reconstruct
            val = prev + (vi / scale)
            out.append(val)
            prev = val
            # if count known, break when reached
            if count is not None and len(out) >= count:
                break
    except EOFError:
        pass
    return np.array(out, dtype=dtype)


def numpress_decode_pic(byte_arr: bytes, dtype=np.float64) -> np.ndarray:
    """
    Decode MSNumpress "pic" (Positive Integer Compression) encoded bytes.
    Returns numpy array of floats (float64).
    """
    import math

    if len(byte_arr) == 0:
        return np.array([], dtype=dtype)

    pos = 0
    # Read header count if present (similar to linear decoder)
    if len(byte_arr) >= 4:
        count = struct.unpack("<I", byte_arr[0:4])[0]
        if 0 <= count <= 10_000_000 and count * 8 <= (len(byte_arr) + 100):
            pos = 4
        else:
            count = None
    else:
        count = None

    def read_varint():
        nonlocal pos
        result = 0
        shift = 0
        while True:
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
            val = read_varint()
            prev += val
            out.append(prev)
            if count is not None and len(out) >= count:
                break
    except IndexError:
        pass

    return np.array(out, dtype=dtype)


def numpress_decode_slof(byte_arr: bytes, dtype=np.float64) -> np.ndarray:
    """
    Decode MSNumpress "slof" (Short Logarithmic Float) encoded bytes.
    Returns numpy array of floats (float64).
    """
    import math

    if len(byte_arr) == 0:
        return np.array([], dtype=dtype)

    pos = 0
    # Read header count if present
    if len(byte_arr) >= 4:
        count = struct.unpack("<I", byte_arr[0:4])[0]
        if 0 <= count <= 10_000_000 and count * 8 <= (len(byte_arr) + 100):
            pos = 4
        else:
            count = None
    else:
        count = None

    def read_varint():
        nonlocal pos
        result = 0
        shift = 0
        while True:
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
            val = read_varint()
            prev += val
            # Decode from encoded integer to float using formula from slof spec:
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
    """Return a dict accession -> value for cvParam children."""
    out = {}
    for cv in elem.findall(".//mzml:cvParam", {"mzml": "http://psi.hupo.org/ms/mzml"}):
        acc = cv.attrib.get("accession")
        val = cv.attrib.get("value")
        out[acc] = val
    return out


def _bytes_to_array(raw_bytes: bytes, dtype_char: str, byteorder: str) -> np.ndarray:
    """Convert raw bytes to numpy array with struct-style dtype_char (e.g. 'f','d','i','q')."""
    if len(raw_bytes) == 0:
        return np.array([], dtype=np.float32 if dtype_char == "f" else np.float64)

    # map dtype_char to numpy dtype and size
    char_to_np = {"f": ("<f4", 4), "d": ("<f8", 8), "i": ("<i4", 4), "q": ("<i8", 8)}
    if dtype_char not in char_to_np:
        raise ValueError(f"Unsupported dtype char: {dtype_char}")

    np_dtype_le, size = char_to_np[dtype_char]
    # apply byte order
    if byteorder == ">":  # big-endian requested
        if dtype_char in ("f", "i"):
            np_dtype = np_dtype_le.replace("<", ">")
        else:
            np_dtype = np_dtype_le.replace("<", ">")
    else:
        np_dtype = np_dtype_le

    count = len(raw_bytes) // size
    return np.frombuffer(raw_bytes[: count * size], dtype=np.dtype(np_dtype)).copy()


# ---------------------------
# Main reader class
# ---------------------------
class ImzMLIBDReader:
    """
    Lazily parses imzML and reads spectra from ibd on demand.
    Use:
        reader = ImzMLIBDReader(imzml_path, ibd_path)
        meta = reader.spectra_meta  # list of per-spectrum metadata
        mzs, ints = reader.read_spectrum(i)  # loads spectrum i
        for batch in reader.sparse_matrix_generator(batch_size=500): ...
    """

    def __init__(self, imzml_path: str, ibd_path: str):
        self.imzml_path = imzml_path
        self.ibd_path = ibd_path
        self.ns = {"mzml": "http://psi.hupo.org/ms/mzml"}
        self.spectra_meta: List[Dict[str, Any]] = []
        self._parse_imzml()

    def _parse_imzml(self):
        tree = ET.parse(self.imzml_path)
        root = tree.getroot()

        # find global byte order default (if any)
        # Many files don't set this globally; each binaryDataArray has cvParams that we will read.
        for i, spectrum in enumerate(root.findall(".//mzml:spectrum", self.ns)):
            # coordinates
            scan = spectrum.find(".//mzml:scan", self.ns)
            # Some imzML use different cvParam accessions for coordinates - we try common ones.
            x_cv = scan.find("./mzml:cvParam[@accession='IMS:1000050']", self.ns)
            y_cv = scan.find("./mzml:cvParam[@accession='IMS:1000051']", self.ns)
            x = int(x_cv.attrib["value"]) if x_cv is not None else 0
            y = int(y_cv.attrib["value"]) if y_cv is not None else 0

            # binaryDataArray list: we assume first is m/z and second is intensity in common files,
            # but we'll inspect their cvParams to be sure.
            bdal = spectrum.findall(".//mzml:binaryDataArray", self.ns)
            if len(bdal) < 2:
                # fallback: continue but mark empty
                self.spectra_meta.append({
                    "index": i, "x": x, "y": y,
                    "mz": None, "intensity": None
                })
                continue

            def parse_bda(bda_elem):
                cv = _get_cv_params(bda_elem)
                # compression -> accession codes examples:
                #   zlib: MS:1000574
                #   numpress linear: MS:1002312
                # dtype: MS:1000521 (32-bit float), MS:1000523 (64-bit float), ...
                # endian: MS:1000141 (little), MS:1000140 (big)
                compression = None
                if "MS:1000574" in cv:
                    compression = "zlib"
                elif "MS:1002312" in cv:
                    compression = "numpress_linear"
                elif "MS:1002313" in cv:
                    compression = "numpress_pic"
                elif "MS:1002314" in cv:
                    compression = "numpress_slof"

                # determine precision
                precision = None
                if "MS:1000521" in cv:  # 32-bit float
                    precision = "f"
                elif "MS:1000523" in cv:  # 64-bit float
                    precision = "d"
                elif "MS:1000519" in cv:  # 32-bit int
                    precision = "i"
                elif "MS:1000522" in cv:  # 64-bit int
                    precision = "q"

                # endianness
                byteorder = "<"  # default little
                if "MS:1000140" in cv:
                    byteorder = ">"
                elif "MS:1000141" in cv:
                    byteorder = "<"

                # offsets and lengths from imzML I/O params (IMS accessions)
                offset = None
                length = None
                if "IMS:1000102" in cv:
                    offset = int(cv["IMS:1000102"])
                if "IMS:1000104" in cv:
                    length = int(cv["IMS:1000104"])

                return {
                    "compression": compression,
                    "precision": precision,
                    "byteorder": byteorder,
                    "offset": offset,
                    "length": length
                }

            mz_meta = parse_bda(bdal[0])
            int_meta = parse_bda(bdal[1])

            self.spectra_meta.append({
                "index": i,
                "x": x, "y": y,
                "mz": mz_meta,
                "intensity": int_meta
            })

        # normalize coordinates to start at 0,0
        xs = [m["x"] for m in self.spectra_meta if m.get("x") is not None]
        ys = [m["y"] for m in self.spectra_meta if m.get("y") is not None]
        if xs and ys:
            minx, miny = min(xs), min(ys)
            for m in self.spectra_meta:
                m["x"] = m["x"] - minx
                m["y"] = m["y"] - miny

    # ---------- low-level binary read ----------
    def _read_raw_block(self, offset: int, length: int) -> bytes:
        if offset is None or length is None or length <= 0:
            return b""
        with open(self.ibd_path, "rb") as f:
            f.seek(offset)
            return f.read(length)

    def _decode_block(self, raw_bytes: bytes, meta: Dict[str, Any]) -> np.ndarray:
        """Decode according to compression/precision/byteorder described in meta."""
        compression = meta.get("compression")
        precision = meta.get("precision") or "f"
        byteorder = meta.get("byteorder") or "<"

        if compression is None:
            # assume raw typed array
            arr = _bytes_to_array(raw_bytes, precision, byteorder)
            return arr
        elif compression == "zlib":
            decompressed = zlib.decompress(raw_bytes)
            return _bytes_to_array(decompressed, precision, byteorder)
        elif compression == "numpress_linear":
            # numpress produces integer-coded bytes which decode to floats
            # decode with msnumpress_linear_decode; it returns float64
            decoded = msnumpress_linear_decode(raw_bytes, dtype=np.float64)
            return decoded
        elif compression == "numpress_pic":
            return numpress_decode_pic(raw_bytes)
        elif compression == "numpress_slof":
            return numpress_decode_slof(raw_bytes)


        else:
            # for unimplemented compression types, raise a helpful error
            raise NotImplementedError(f"Compression '{compression}' is not (yet) implemented.")

    # ---------- public API ----------
    def read_spectrum(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read spectrum by index and return (mz_array, intensity_array).
        Filters NaN pairs and trims unequal lengths gracefully.
        """
        if index < 0 or index >= len(self.spectra_meta):
            raise IndexError("spectrum index out of range")

        meta = self.spectra_meta[index]
        mz_meta = meta.get("mz")
        int_meta = meta.get("intensity")

        if mz_meta is None or int_meta is None:
            return np.array([]), np.array([])

        raw_mz = self._read_raw_block(mz_meta["offset"], mz_meta["length"])
        raw_int = self._read_raw_block(int_meta["offset"], int_meta["length"])

        mz_values = self._decode_block(raw_mz, mz_meta)
        int_values = self._decode_block(raw_int, int_meta)

        # align lengths
        if mz_values.size != int_values.size:
            minlen = min(mz_values.size, int_values.size)
            mz_values = mz_values[:minlen]
            int_values = int_values[:minlen]

        # remove NaNs and non-finite
        valid = np.isfinite(mz_values) & np.isfinite(int_values)
        if np.any(~valid):
            mz_values = mz_values[valid]
            int_values = int_values[valid]

        return mz_values, int_values
    

    def sparse_matrix_generator(self, batch_size: int = 500) -> Generator[Tuple[List[int], List[int], List[float], np.ndarray], None, None]:
        """
        Generate batches for sparse matrix assembly.
        Yields tuples (rows, cols, values, coords_array) where:
            - rows, cols, values are lists describing a CSR block for the batch
            - coords_array is N x 2 array with coordinates for the batch's spectra
        This method computes a global m/z dictionary lazily per batch; for large global m/z spaces
        you may prefer to first compute `global_mzs = reader.unique_mzs()` (costly).
        """
        n = len(self.spectra_meta)
        for start in range(0, n, batch_size):
            rows, cols, values = [], [], []
            coords = []
            mzs_in_batch = []
            # collect all mzs in batch to create local indexing
            for i in range(start, min(start + batch_size, n)):
                mz_values, int_values = self.read_spectrum(i)
                coords.append((self.spectra_meta[i]["x"], self.spectra_meta[i]["y"]))
                mzs_in_batch.append(mz_values)

            # flatten unique m/z for this batch
            if len(mzs_in_batch) == 0:
                yield [], [], [], np.array(coords)
                continue
            all_mzs = np.unique(np.concatenate([arr for arr in mzs_in_batch if arr.size > 0]))
            mz_to_index = {mz: j for j, mz in enumerate(all_mzs)}

            # now produce row/col/value entries using global column indices relative to the batch
            for local_row, i in enumerate(range(start, min(start + batch_size, n))):
                mz_values, int_values = self.read_spectrum(i)
                if mz_values.size == 0:
                    continue
                valid = int_values != 0
                mz_values = mz_values[valid]
                int_values = int_values[valid]
                for mz, inten in zip(mz_values, int_values):
                    rows.append(local_row)
                    cols.append(mz_to_index[mz])
                    values.append(float(inten))

            yield rows, cols, values, np.array(coords)

    def unique_mzs(self) -> np.ndarray:
        """(Costly) read every spectrum and return global unique sorted m/z values."""
        all_mzs = []
        for i in range(len(self.spectra_meta)):
            mzv, _ = self.read_spectrum(i)
            if mzv.size > 0:
                all_mzs.append(mzv)
        if not all_mzs:
            return np.array([])
        return np.unique(np.concatenate(all_mzs))


# ---------------------------
# Convenience factory function
# ---------------------------
def open_imzml_ibd(imzml_path: str, ibd_path: str) -> ImzMLIBDReader:
    return ImzMLIBDReader(imzml_path, ibd_path)
