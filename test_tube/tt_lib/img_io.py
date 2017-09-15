"""
Module to save a numpy array as an image
Taken from SO and modified a bit to work with this app
https://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image
"""


def save_as_png(array, filename):
    if any([len(row) != len(array[0]) for row in array]):
        raise ValueError('Array should have elements of equal size')
    data = _write_png(bytearray(array[::-1]), len(array[0]), len(array))
    f = open(filename, 'wb')
    f.write(data)
    f.close()


def _write_png(buf, width, height):
    """
    Taken from here:
    https://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image

    buf: must be bytes or a bytearray in Python3.x,
        a regular string in Python2.x.
    """
    import zlib, struct

    # reverse the vertical line order and add null bytes at the start
    width_byte_4 = width * 4
    raw_data = b''.join(b'\x00' + buf[span:span + width_byte_4]
                        for span in range((height - 1) * width_byte_4, -1, - width_byte_4))

    def png_pack(png_tag, data):
        chunk_head = png_tag + data
        return (struct.pack("!I", len(data)) +
                chunk_head +
                struct.pack("!I", 0xFFFFFFFF & zlib.crc32(chunk_head)))

    return b''.join([
        b'\x89PNG\r\n\x1a\n',
        png_pack(b'IHDR', struct.pack("!2I5B", width, height, 8, 6, 0, 0, 0)),
        png_pack(b'IDAT', zlib.compress(raw_data, 9)),
        png_pack(b'IEND', b'')])
