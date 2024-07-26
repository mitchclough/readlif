from __future__ import annotations

import io
import os
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Generator
from itertools import compress
from typing import Any, Generic, Literal, NamedTuple, TypeAlias, TypeVar

from PIL import Image

from ._helpers import (
    FileDescriptorOrPath,
    check_magic,
    check_mem,
    check_truncated,
    get_len,
    read_u32,
    read_u64,
)

FileDescriptorOrPathOrReader: TypeAlias = FileDescriptorOrPath | io.BufferedReader

T = TypeVar("T", int, float, str)


class Dims(NamedTuple, Generic[T]):
    """
    Class with integer values for all of the possible dimensions from a lif file.
    """

    x: T
    """the value of interest for the x dimension"""
    y: T
    """the value of interest for the y dimension"""
    z: T
    """the value of interest for the z dimension"""
    t: T
    """the value of interest for the t dimension"""
    wl_em: T
    """the value of interest for the emission wavelength dimension"""
    wl_ex: T
    """the value of interest for the excitation wavelength dimension"""
    m: T
    """the value of interest for the mosaic tile dimension"""

    @staticmethod
    def make_int(
        x: int = 0,
        y: int = 0,
        z: int = 0,
        t: int = 0,
        wl_em: int = 0,
        wl_ex: int = 0,
        m: int = 0,
    ) -> Dims[int]:
        """
        Factory function to create a Dims[int] instance. Allows to only specify
        dimensions of interest and leave the others to a default value.

        Args:
            x: value for the X dimension
            y: value for the Y dimension
            z: value for the Z dimension
            t: value for the T dimension
            wl_em: value for the Wl_Em dimension
            wl_ex: value for the Wl_Ex dimension
            m: value for the M dimension
        """
        return Dims(x=x, y=y, z=z, t=t, wl_em=wl_em, wl_ex=wl_ex, m=m)

    @staticmethod
    def make_float(
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        t: float = 0.0,
        wl_em: float = 0.0,
        wl_ex: float = 0.0,
        m: float = 0.0,
    ) -> Dims[float]:
        """
        Factory function to create a Dims[float] instance. Allows to only specify
        dimensions of interest and leave the others to a default value.

        Args:
            x: value for the X dimension
            y: value for the Y dimension
            z: value for the Z dimension
            t: value for the T dimension
            wl_em: value for the Wl_Em dimension
            wl_ex: value for the Wl_Ex dimension
            m: value for the M dimension
        """
        return Dims(x=x, y=y, z=z, t=t, wl_em=wl_em, wl_ex=wl_ex, m=m)

    @staticmethod
    def make_str(
        x: str = "",
        y: str = "",
        z: str = "",
        t: str = "",
        wl_em: str = "",
        wl_ex: str = "",
        m: str = "",
    ) -> Dims[str]:
        """
        Factory function to create a Dims[str] instance. Allows to only specify
        dimensions of interest and leave the others to a default value.

        Args:
            x: value for the X dimension
            y: value for the Y dimension
            z: value for the Z dimension
            t: value for the T dimension
            wl_em: value for the Wl_Em dimension
            wl_ex: value for the Wl_Ex dimension
            m: value for the M dimension
        """
        return Dims(x=x, y=y, z=z, t=t, wl_em=wl_em, wl_ex=wl_ex, m=m)


class Tile(NamedTuple):
    """
    Class with positions for a mosaic tile.
    """

    field: tuple[int, int]
    """(x, y) field index"""
    pos: tuple[float, float]
    """(x, y) tile position"""


class ImageInfo(NamedTuple):
    """
    Class with metadata and info about the image dataset.
    """

    subfile_path: str
    """path to the image in the lif file"""
    name: str
    """name of the image"""
    dim_sizes: Dims[int]
    """number of elements in each dimension for the image"""
    dim_byte_incs: Dims[int]
    """number of bytes to increment in the raw data to increment this dimension by 1"""
    dim_scales: Dims[float]
    """
    scale of each dimension in SI units corresponding to the dimension
    (e.g. px / m, frames / s)
    """
    channels: int
    """number of channels in the image data"""
    chan_byte_incs: tuple[int, ...]
    """number of bytes to increment in the raw data to increment the channel by 1"""
    chan_bit_depths: tuple[int, ...]
    """bit depth for each channel in the image data"""
    mosaic_positions: tuple[Tile, ...]
    """
    If the image is a mosaic (tiled), this contains a list of Tile objects which
    contains the field and position values for the tile
    """
    settings: dict[str, str]
    """
    ATLConfocalSettingDefinition (if it exists), which contains values like
    NumericalAperture and Magnification.
    """


class LifImage:
    """
    This should not be called directly. This should be generated while calling
    get_image or get_iter_image from a LifFile object.

    Attributes:
        filename: The LIF file containing this image.
        offsets: Byte position offsets for each image.
        image_info: Class containing most of the information
            needed to read the image from the file.
        display_dims: The first two dimensions of the lif file. This is used to decide
            what dimensions are returned in a 2D plane.
        bpp: Bytes per pixel in the image data.
    """

    def __init__(
        self,
        filename: FileDescriptorOrPathOrReader,
        offsets: tuple[int, int],
        image_info: ImageInfo,
    ):
        """
        Initializes the LifImage instance.

        Args:
            filename: The LIF file containing this image.
            offsets: Byte position offsets for each image.
            image_info: Class containing most of the information needed to read the
                image from the file.
        """
        self.filename: FileDescriptorOrPathOrReader = filename
        self.offsets: tuple[int, int] = offsets

        self.image_info = image_info

        display_dims: list[str] = []
        n_dims = 0
        for i, d in enumerate(self.image_info.dim_sizes):
            if d > 0:
                display_dims.append(Dims._fields[i])
                n_dims += 1

            if n_dims >= 2:
                break
        self.display_dims: tuple[str, str] = (display_dims[0], display_dims[1])

        dim_byte_incs_list = [b for b in self.image_info.dim_byte_incs if b >= 0]
        if min(self.image_info.chan_byte_incs) == 0:
            self.bpp = min(dim_byte_incs_list)
        elif min(dim_byte_incs_list) == 0:
            self.bpp = min(self.image_info.chan_byte_incs)
        else:
            msg = "cannot determine number of bytes per pixel"
            raise ValueError(msg)

    def __repr__(self) -> str:
        return repr(f"LifImage object with dimensions: {self.image_info.dim_sizes}")

    def get_plane(
        self,
        display_dims: tuple[str, str] | None = None,
        c: int = 0,
        requested_dims: Dims[int] | None = None,
    ) -> Image.Image:
        """
        Gets the specified frame from image.

        Args:
            display_dims: Two value tuple ("X", "Y") specifying the two dimension plane
                to return. This will default to the first two dimensions in the LifFile,
                specified by LifImage.display_dims
            c: channel
            requested_dims: Object containing the dimension indexes to return.

        Returns:
            Pillow Image object
        """
        if requested_dims is None:
            requested_dims = Dims.make_int()

        if display_dims is None:
            display_dims = self.display_dims
        elif not isinstance(display_dims, tuple) or len(display_dims) != 2:
            msg = "display_dims must be a two value tuple"
            raise ValueError(msg)

        if c + 1 > self.image_info.channels:
            msg = (
                f"Requested Channel {c} but image only has {self.image_info.channels} "
                "channels"
            )
            raise ValueError(msg)

        # Check if any of the dims exceeds what is in the image
        for i, d in enumerate(requested_dims):
            if d != 0 and (
                self.image_info.dim_sizes[i] == 0 or d > self.image_info.dim_sizes[i]
            ):
                msg = (
                    f"Requested frame in dimension {requested_dims._fields[i]} "
                    "doesn't exist"
                )
                raise ValueError(msg)

        if isinstance(self.filename, (int, str, bytes, os.PathLike)):
            image = open(self.filename, "rb")  # noqa: SIM115
        elif isinstance(self.filename, io.IOBase):
            image = self.filename
        else:
            msg = (
                "expected str, bytes, os.PathLike, or io.IOBase, not "
                f"{type(self.filename)}"
            )
            raise TypeError(msg)

        # Start at the beginning of the specified image
        image.seek(self.offsets[0])
        data = b""

        plane_dims = (
            Dims._fields.index(display_dims[0].lower()),
            Dims._fields.index(display_dims[1].lower()),
        )
        max_off_x = (
            self.image_info.dim_byte_incs[plane_dims[0]]
            * self.image_info.dim_sizes[plane_dims[0]]
        )
        increment_x = self.image_info.dim_byte_incs[plane_dims[0]]
        display_x = range(0, max_off_x, increment_x)
        max_off_y = (
            self.image_info.dim_byte_incs[plane_dims[1]]
            * self.image_info.dim_sizes[plane_dims[1]]
        )
        increment_y = self.image_info.dim_byte_incs[plane_dims[1]]
        display_y = range(0, max_off_y, increment_y)

        # go to starting position for the channel and requested_dims based on the bytes
        # offset from lif metadata
        start_pos = 0
        for i, d in enumerate(requested_dims):
            start_pos += self.image_info.dim_byte_incs[i] * d
        start_pos += self.image_info.chan_byte_incs[c]

        # Speedup for the common case where the plane_dims are the first two dims
        #  i.e. reading the number of image pixels times the number of bytes per pixel
        # gives us the correct data
        contains_bpp = self.image_info.dim_byte_incs[plane_dims[0]] == self.bpp
        contains_bpp_times_other = (
            self.image_info.dim_byte_incs[plane_dims[0]]
            == self.bpp * self.image_info.dim_byte_incs[plane_dims[1]]
        )
        # Quickest case where we can just read bpp * nx * ny bytes from the file and get
        # our image
        if contains_bpp and contains_bpp_times_other:
            # Define the size of the plane to return
            read_len = (
                self.image_info.dim_sizes[plane_dims[0]]
                * self.image_info.dim_sizes[plane_dims[1]]
                * self.bpp
            )
            if self.offsets[1] == 0:
                data = data + b"\00" * read_len
            else:
                image.seek(self.offsets[0] + start_pos)
                data = data + image.read(read_len)
        # Quicker case where we can't read the whole image at once but can read in one
        # line at a time
        elif contains_bpp:
            read_len = self.image_info.dim_sizes[plane_dims[0]] * self.bpp

            for pos in display_y:
                px_pos = start_pos + pos
                if self.offsets[1] == 0:
                    data = data + b"\00" * read_len
                else:
                    image.seek(self.offsets[0] + px_pos)
                    data = data + image.read(read_len)
        # Handle the less common case, where the plane_dims are arbitrary
        else:
            for pos_y in display_y:
                for pos_x in display_x:
                    px_pos = start_pos + pos_x + pos_y
                    if self.offsets[1] == 0:
                        data = data + b"\00" * self.bpp
                    else:
                        image.seek(self.offsets[0] + px_pos)
                        data = data + image.read(self.bpp)

        if isinstance(self.filename, (int, str, bytes, os.PathLike)):
            image.close()

        # LIF files can be either 8-bit of 16-bit.
        # Because of how the image is read in, all of the raw
        # data is already in 'data', we just need to tell Pillow
        # how to set the bit depth
        # 'L' is 8-bit, 'I;16' is 16 bit

        # len(data) is the number of bytes (8-bit)
        # However, it is safer to let the lif file tell us the resolution
        if self.image_info.chan_bit_depths[0] == 8:
            return Image.frombytes(
                "L",
                (
                    self.image_info.dim_sizes[plane_dims[0]],
                    self.image_info.dim_sizes[plane_dims[1]],
                ),
                data,
            )

        if self.image_info.chan_bit_depths[0] <= 16:
            return Image.frombytes(
                "I;16",
                (
                    self.image_info.dim_sizes[plane_dims[0]],
                    self.image_info.dim_sizes[plane_dims[1]],
                ),
                data,
            )

        msg = "Unknown bit-depth, please submit a bug report on Github"
        raise ValueError(msg)

    def get_frame(self, z: int = 0, t: int = 0, c: int = 0, m: int = 0) -> Image.Image:
        """
        Gets the specified frame (z, t, c, m) from image.

        Args:
            z: z position
            t: time point
            c: channel
            m: mosaic image

        Returns:
            Pillow Image object
        """
        if self.display_dims != ("x", "y"):
            msg = (
                "Atypical imaging experiment, please use "
                "get_plane() instead of get_frame()"
            )
            raise ValueError(msg)

        if z != 0 and z >= self.image_info.dim_sizes.z:
            msg = "Requested Z frame doesn't exist."
            raise ValueError(msg)

        if t != 0 and t >= self.image_info.dim_sizes.t:
            msg = "Requested T frame doesn't exist."
            raise ValueError(msg)

        if c >= self.image_info.channels:
            msg = "Requested channel doesn't exist."
            raise ValueError(msg)

        if m != 0 and m >= self.image_info.dim_sizes.m:
            msg = "Requested mosaic image doesn't exist."
            raise ValueError(msg)

        return self.get_plane(
            display_dims=("X", "Y"), c=c, requested_dims=Dims.make_int(z=z, t=t, m=m)
        )

    def get_iter_t(
        self, z: int = 0, c: int = 0, m: int = 0
    ) -> Generator[Image.Image, Any, None]:
        """
        Returns an iterator over time t at position z and channel c.

        Args:
            z: z position
            c: channel
            m: mosaic image

        Returns:
            Iterator of Pillow Image objects
        """
        for t in range(self.image_info.dim_sizes.t):
            yield self.get_frame(z=z, t=t, c=c, m=m)

    def get_iter_c(
        self, z: int = 0, t: int = 0, m: int = 0
    ) -> Generator[Image.Image, Any, None]:
        """
        Returns an iterator over the channels at time t and position z.

        Args:
            z: z position
            t: time point
            m: mosaic image

        Returns:
            Iterator of Pillow Image objects
        """
        for c in range(self.image_info.channels):
            yield self.get_frame(z=z, t=t, c=c, m=m)

    def get_iter_z(
        self, t: int = 0, c: int = 0, m: int = 0
    ) -> Generator[Image.Image, Any, None]:
        """
        Returns an iterator over the z series of time t and channel c.

        Args:
            t: time point
            c: channel
            m: mosaic image

        Returns:
            Iterator of Pillow Image objects
        """
        for z in range(self.image_info.dim_sizes.z):
            yield self.get_frame(z=z, t=t, c=c, m=m)

    def get_iter_m(
        self, z: int = 0, t: int = 0, c: int = 0
    ) -> Generator[Image.Image, Any, None]:
        """
        Returns an iterator over the z series of time t and channel c.

        Args:
            t: time point
            c: channel
            z: z position

        Returns:
            Iterator of Pillow Image objects
        """
        for m in range(self.image_info.dim_sizes.m):
            yield self.get_frame(z=z, t=t, c=c, m=m)


class LifFile:
    """
    Given a path or buffer to a lif file, returns objects containing
    the image and data.

    This is based on the java openmicroscopy bioformats lif reading code
    that is here: https://github.com/openmicroscopy/bioformats/blob/master/components/formats-gpl/src/loci/formats/in/LIFReader.java

    Attributes:
        filename: File descriptor, path, or buffered reader to the lif file.
        xml_header: The LIF xml header with tons of data
        xml_root: ElementTree XML representation
        offsets: Byte positions of the files
        num_images: Number of images.
        image_list: List of readlif.reader.ImageInfo objects with one for each image.

    Example:
        >>> from readlif.reader import LifFile
        >>> new = LifFile('./path/to/file.lif')

        >>> for image in new.get_iter_image():
        >>>     for frame in image.get_iter_t():
        >>>         frame.image_info['name']
        >>>         # do stuff

        >>> # For non-xy imaging experiments
        >>> img_0 = new.get_image(0)
        >>> for i in range(0, img_0.dims_n[4]):
        >>>     plane = img_0.get_plane(requested_dims = {4: i})
    """

    def _recursive_image_find(
        self,
        tree: ET.Element,
        return_list: list[ImageInfo | Literal[False]] | None = None,
        path: str = "",
    ) -> list[ImageInfo | Literal[False]]:
        """Creates list of images by parsing the XML header recursively"""

        if return_list is None:
            return_list = []

        children = tree.findall("./Children/Element")
        if len(children) < 1:  # Fix for 'first round'
            children = tree.findall("./Element")
        for item in children:
            folder_name = item.get("Name", "")
            # Grab the .lif filename name on the first execution
            appended_path = folder_name if path == "" else path + "/" + folder_name
            # This finds empty folders
            has_sub_children = len(item.findall("./Children/Element/Data")) > 0

            is_image = len(item.findall("./Data/Image")) > 0

            # Check to see if the Memblock idnetified in the XML actually has a size,
            # otherwise it won't have an offset
            memory_element = item.find("./Memory")
            if memory_element is None:
                msg = "No memory element was found in the XML metadata"
                raise ValueError(msg)
            if int(memory_element.get("Size", 0)) > 0:
                if is_image:
                    # If additional XML data extraction is needed, add it here.

                    # Find the dimensions, get them in order
                    dims = item.findall("./Data/Image/ImageDescription/" "Dimensions/")

                    dims_dict: dict[int, int] = {}
                    dims_bytes: dict[int, int] = {}
                    dims_scale: dict[int, float] = {}
                    for d in dims:
                        dim_id = int(d.get("DimID", 0))
                        dims_dict[dim_id] = int(d.get("NumberOfElements", 0))
                        dims_bytes[dim_id] = int(d.get("BytesInc", -1))
                        dim_len = float(d.get("Length", 0))
                        if dim_len > 0:
                            dims_scale[dim_id] = (dims_dict[dim_id] - 1) / dim_len
                        else:
                            dims_scale[dim_id] = 0

                    # Known LIF dims:
                    # 1: x
                    # 2: y
                    # 3: z
                    # 4: t
                    # 5: detection wavelength
                    # 6: Unknown
                    # 7: Unknown
                    # 8: Unknown
                    # 9: illumination wavelength
                    # 10: Mosaic tile
                    dim_sizes = Dims.make_int(
                        x=dims_dict.get(1, 0),
                        y=dims_dict.get(2, 0),
                        z=dims_dict.get(3, 0),
                        t=dims_dict.get(4, 0),
                        wl_em=dims_dict.get(5, 0),
                        wl_ex=dims_dict.get(9, 0),
                        m=dims_dict.get(10, 0),
                    )

                    dim_byte_incs = Dims.make_int(
                        x=dims_bytes.get(1, -1),
                        y=dims_bytes.get(2, -1),
                        z=dims_bytes.get(3, -1),
                        t=dims_bytes.get(4, -1),
                        wl_em=dims_bytes.get(5, -1),
                        wl_ex=dims_bytes.get(9, -1),
                        m=dims_bytes.get(10, -1),
                    )

                    dim_scales = Dims.make_float(
                        x=dims_scale.get(1, 0.0),
                        y=dims_scale.get(2, 0.0),
                        z=dims_scale.get(3, 0.0),
                        t=dims_scale.get(4, 0.0),
                        wl_em=dims_scale.get(5, 0.0),
                        wl_ex=dims_scale.get(9, 0.0),
                        m=dims_scale.get(10, 0.0),
                    )

                    # Determine number of channels
                    channel_list = item.findall(
                        "./Data/Image/ImageDescription/Channels/ChannelDescription"
                    )
                    chan_byte_incs = tuple(
                        int(c.get("BytesInc", -1)) for c in channel_list
                    )
                    n_channels = int(len(channel_list))
                    # Iterate over each channel, get the resolution
                    chan_bit_depths = tuple(
                        int(c.get("Resolution", 0)) for c in channel_list
                    )

                    # Get the position data if the image is tiled
                    if dim_sizes.m > 1:
                        tiles = tuple(
                            Tile(
                                field=(
                                    int(tile.get("FieldX", 0)),
                                    int(tile.get("FieldY", 0)),
                                ),
                                pos=(
                                    float(tile.get("PosX", 0)),
                                    float(tile.get("PosY", 0)),
                                ),
                            )
                            for tile in item.findall("./Data/Image/Attachment/Tile")
                        )
                    else:
                        tiles = ()

                    settings_list = item.find(
                        "./Data/Image/Attachment/ATLConfocalSettingDefinition"
                    )
                    settings = settings_list.attrib if settings_list is not None else {}

                    tmp_name = path + "/" + item.get("Name", "")
                    first_slash_idx = tmp_name.find("/")
                    name = tmp_name[first_slash_idx + 1 :]
                    image_info = ImageInfo(
                        subfile_path=path + "/",
                        name=name,
                        dim_sizes=dim_sizes,
                        dim_byte_incs=dim_byte_incs,
                        dim_scales=dim_scales,
                        channels=n_channels,
                        chan_byte_incs=chan_byte_incs,
                        chan_bit_depths=chan_bit_depths,
                        mosaic_positions=tile_list,
                        settings=settings,
                    )

                    return_list.append(image_info)
                else:
                    return_list.append(False)

            # An image can have sub_children, it is not mutually exclusive
            if has_sub_children:
                self._recursive_image_find(item, return_list, appended_path)

        return return_list

    def __init__(self, filename: FileDescriptorOrPathOrReader):
        """
        Initialize a LifFile instance.

        Args:
            filename: File descriptor, path, or buffered reader to the lif file.
        """
        self.filename = filename

        if isinstance(filename, (int, str, bytes, os.PathLike)):
            f = open(filename, "rb")  # noqa: SIM115
        elif isinstance(filename, io.BufferedReader):
            f = filename
        else:
            msg = (
                f"expected str, bytes, os.PathLike, or io.IOBase, not {type(filename)}"
            )
            raise TypeError(msg)
        f_len = get_len(f)

        check_magic(f)  # read 4 byte, check for magic bytes
        f.seek(8)
        check_mem(f)  # read 1 byte, check for memory byte

        header_len = read_u32(f)  # length of the xml header
        self.xml_header = f.read(header_len * 2).decode("utf-16")
        self.xml_root = ET.fromstring(self.xml_header)

        self.offsets: list[tuple[int, int]] = []
        truncated = False
        while f.tell() < f_len:
            try:
                # To find offsets, read magic byte
                check_magic(f)  # read 4 byte, check for magic bytes
                f.seek(4, 1)
                check_mem(f)  # read 1 byte, check for memory byte

                block_len = read_u32(f)

                # Not sure if this works, as I don't have a file to test it on
                # This is based on the OpenMicroscopy LIF reader written in in java
                if not check_mem(f, True):
                    f.seek(-5, 1)
                    block_len = read_u64(f)
                    check_mem(f)

                description_len = read_u32(f) * 2

                if block_len > 0:
                    self.offsets.append((f.tell() + description_len, block_len))

                f.seek(description_len + block_len, 1)

            except ValueError:
                if check_truncated(f):
                    truncation_begin = f.tell()
                    warnings.warn(
                        "LIF file is likely truncated. Be advised, "
                        "it appears that some images are blank. ",
                        UserWarning,
                        stacklevel=2,
                    )
                    truncated = True
                    f.seek(0, 2)

                else:
                    raise

        if isinstance(filename, (str, bytes, os.PathLike)):
            f.close()

        tmp_image_list = self._recursive_image_find(self.xml_root)

        # If the image is truncated we need to manually add the offsets because
        # the LIF magic bytes aren't present to guide the location.
        if truncated:
            num_truncated = len(tmp_image_list) - len(self.offsets)
            for _ in range(num_truncated):
                # In the special case of a truncation,
                # append an offset with length zero.
                # This will be taken care of later when the images are retrieved.
                self.offsets.append((truncation_begin, 0))

        # Fix for new LASX version
        num_images = sum(1 if x else 0 for x in tmp_image_list)
        if len(self.offsets) > num_images and False in tmp_image_list:
            self.offsets = list(compress(self.offsets, tmp_image_list))

        self.image_list = [x for x in tmp_image_list if x]

        if len(self.image_list) != len(self.offsets) and not truncated:
            msg = (
                "Number of images is not equal to number of "
                "offsets, and this file does not appear to "
                "be truncated. Something has gone wrong."
            )
            raise ValueError(msg)

        self.num_images = len(self.image_list)

    def __repr__(self) -> str:
        if self.num_images == 1:
            return repr("LifFile object with " + str(self.num_images) + " image")

        return repr("LifFile object with " + str(self.num_images) + " images")

    def get_image(self, img_n: int = 0) -> LifImage:
        """
        Specify the image number, and this returns a LifImage object
        of that image.

        Args:
            img_n (int): Image number to retrieve

        Returns:
            LifImage object with specified image
        """
        if img_n >= len(self.image_list):
            msg = "There are not that many images!"
            raise ValueError(msg)
        offsets = self.offsets[img_n]
        image_info = self.image_list[img_n]
        return LifImage(self.filename, offsets, image_info)

    def get_iter_image(self, img_n: int = 0) -> Generator[LifImage, Any, None]:
        """
        Returns an iterator of LifImage objects in the lif file.

        Args:
            img_n (int): Image to start iteration at

        Returns:
            Iterator of LifImage objects.
        """
        for i in range(img_n, len(self.image_list)):
            yield LifImage(self.filename, self.offsets[i], self.image_list[i])
