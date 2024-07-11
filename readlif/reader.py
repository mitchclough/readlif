from __future__ import annotations

import io
import os
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Generator
from itertools import compress
from typing import Any, NamedTuple, TypeAlias

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


class Dims(NamedTuple):
    x: int
    y: int
    z: int
    t: int
    m: int


class LifImage:
    """
    This should not be called directly. This should be generated while calling
    get_image or get_iter_image from a LifFile object.

    Attributes:
        path (str): path / name of the image
        dims (tuple): (x, y, z, t, m)
        display_dims (tuple): The first two dimensions of the lif file.
            This is used to decide what dimensions are returned in a 2D plane.
        dims_n (dict): {0: length, 1: length, 2: length, n: length}

            For atypical imaging experiments, i.e. those not simple photos
            of XY frames, this attribute will be more useful than `dims`.
            This attribute will hold a dictionary with the length of each
            dimension, in the order it is referenced in the .lif file.

            Currently, only some of the 10 possible dimensions are used / known:

            - 1: x-axis

            - 2: y-axis

            - 3: z-axis

            - 4: time

            - 5: detection wavelength

            - 6-8: Unknown

            - 9: illumination wavelength

            - 10: mosaic tile

        name (str): image name
        offsets (list): Byte position offsets for each image.
        filename (str, bytes, os.PathLike, io.IOBase): The name of the LIF file
            being read
        channels (int): Number of channels in the image
        nz (int): number of 'z' frames

            Note, it is recommended to use `dims.z` instead. However, this will
            be kept for compatibility.
        nt (int): number of 't' frames

            Note, it is recommended to use `dims.t` instead. However, this will
            be kept for compatibility.
        scale (tuple): (scale_x, scale_y, scale_z, scale_t).

            Conversion factor: px/µm for x, y and z; images/frame for t. For
            time, this is the duration for the entire image acquisition.
        scale_n (dict): {1: scale_x, 2: scale_y, 3: scale_z, 4: scale_t}.

            Conversion factor: px/µm for x, y and z; images/sec for t. Related
            to `dims_n` above.
        bit_depth (tuple): A tuple of ints that indicates the bit depth of
            each channel in the image.
        mosaic_position (list): If the image is a mosaic (tiled), this contains
            a list of tuples with four values: `(FieldX, FieldY, PosX, PosY)`.
            The length of this list is equal to the number of tiles.
        settings (dict): ATLConfocalSettingDefinition (if it exists), which contains
            values like NumericalAperture and Magnification.
        info (dict): Direct access to data dict from LifFile, this is most
            useful for debugging. These are values pulled from the Leica XML.
    """

    def __init__(
        self,
        image_info: dict[str, Any],
        offsets: tuple[int, int],
        filename: FileDescriptorOrPathOrReader,
    ):
        self.dims: Dims = image_info["dims"]
        self.display_dims: tuple[int, int] = image_info["display_dims"]
        self.dims_n: dict[int, int] = image_info["dims_n"]
        self.scale_n: dict[int, float | None] = image_info["scale_n"]
        self.path: str = image_info["path"]
        self.offsets: tuple[int, int] = offsets
        self.info: dict[str, Any] = image_info
        self.filename: FileDescriptorOrPathOrReader = filename
        self.name: str = image_info["name"]
        self.channels: int = image_info["channels"]
        self.nz: int = int(image_info["dims"].z)
        self.nt: int = int(image_info["dims"].t)
        self.scale: tuple[int, int, int, int] = image_info["scale"]
        self.bit_depth: tuple[int, ...] = image_info["bit_depth"]
        self.mosaic_position: list[tuple[int, int, float, float]] = image_info[
            "mosaic_position"
        ]
        self.n_mosaic: int = self.dims.m
        self.settings: dict[str, str] = image_info["settings"]
        self.dims_bytes: dict[int, int] = image_info["dims_bytes"]
        self.channel_bytes: list[int] = image_info["channels_bytes"]

        dims_bytes_list = list(image_info["dims_bytes"].values())
        if min(self.channel_bytes) == 0:
            self.bpp = min(dims_bytes_list)
        elif min(dims_bytes_list) == 0:
            self.bpp = min(self.channel_bytes)
        else:
            msg = "cannot determine number of bytes per pixel"
            raise ValueError(msg)

    def __repr__(self) -> str:
        return repr("LifImage object with dimensions: " + str(self.dims))

    def get_plane(
        self,
        display_dims: tuple[int, int] | None = None,
        c: int = 0,
        requested_dims: dict[int, int] | None = None,
    ) -> Image.Image:
        """
        Gets the specified frame from image.

        Known issue: LASX Navigator saves channel as the second channel, this reader
        will fail in that case.

        Args:
            display_dims (tuple): Two value tuple (1, 2) specifying the
                two dimension plane to return. This will default to the first
                two dimensions in the LifFile, specified by LifFile.display_dims
            c (int): channel
            requested_dims (dict): Dictionary indicating the item to be returned,
                as described by a numerical dictionary, ex: {3: 0, 4: 1}

        Returns:
            Pillow Image object
        """
        if requested_dims is None:
            requested_dims = {}

        if display_dims is None:
            display_dims = self.display_dims
        elif type(display_dims) is not tuple or len(display_dims) != 2:
            msg = "display_dims must be a two value tuple"
            raise ValueError(msg)

        if requested_dims.keys() in display_dims:
            warnings.warn(
                "One or more of the display_dims is in the "
                "requested_dims dictionary. Currently this has no "
                "effect. All data from the display_dims will be "
                "returned.",
                stacklevel=2,
            )

        if display_dims != self.display_dims:
            msg = "Arbitrary dimensions are not yet supported"
            raise NotImplementedError(msg)

        # Set all requested dims to 0:
        for i in range(1, 11):
            requested_dims[i] = int(requested_dims.get(i, 0))

        if c + 1 > self.channels:
            msg = f"Requested Channel {c} but image only has {self.channels} channels"
            raise ValueError(msg)

        # Check if any of the dims exceeds what is in the image
        for key in requested_dims:
            if requested_dims[key] != 0 and (
                self.dims_n.get(key) is None or requested_dims[key] > self.dims_n[key]
            ):
                msg = f"Requested frame in dimension {key} doesn't exist"
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

        max_off_x = self.dims_bytes[display_dims[0]] * self.dims_n[display_dims[0]]
        increment_x = self.dims_bytes[display_dims[0]]
        display_x = range(0, max_off_x, increment_x)
        max_off_y = self.dims_bytes[display_dims[1]] * self.dims_n[display_dims[1]]
        increment_y = self.dims_bytes[display_dims[1]]
        display_y = range(0, max_off_y, increment_y)

        # go to starting position for the channel and requested_dims based on the bytes
        # offset from lif metadata
        start_pos = 0
        for key in requested_dims:
            start_pos += self.dims_bytes.get(key, 0) * requested_dims[key]
        start_pos += self.channel_bytes[c]

        # Speedup for the common case where the display_dims are the first two dims
        #  i.e. reading the number of image pixels times the number of bytes per pixel
        # gives us the correct data
        contains_bpp = self.dims_bytes[display_dims[0]] == self.bpp
        contains_bpp_times_other = (
            self.dims_bytes[display_dims[0]]
            == self.bpp * self.dims_bytes[display_dims[1]]
        )
        # Quickest case where we can just read bpp * nx * ny bytes from the file and get
        # our image
        if contains_bpp and contains_bpp_times_other:
            # Define the size of the plane to return
            read_len = (
                self.dims_n[display_dims[0]] * self.dims_n[display_dims[1]] * self.bpp
            )
            if self.offsets[1] == 0:
                data = data + b"\00" * read_len
            else:
                image.seek(self.offsets[0] + start_pos)
                data = data + image.read(read_len)
        # Quicker case where we can't read the whole image at once but can read in one
        # line at a time
        elif contains_bpp:
            read_len = self.dims_n[display_dims[0]] * self.bpp

            for pos in display_y:
                px_pos = start_pos + pos
                if self.offsets[1] == 0:
                    data = data + b"\00" * read_len
                else:
                    image.seek(self.offsets[0] + px_pos)
                    data = data + image.read(read_len)
        # Handle the less common case, where the display_dims are arbitrary
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
        if self.bit_depth[0] == 8:
            return Image.frombytes(
                "L", (self.dims_n[display_dims[0]], self.dims_n[display_dims[1]]), data
            )

        if self.bit_depth[0] <= 16:
            return Image.frombytes(
                "I;16",
                (self.dims_n[display_dims[0]], self.dims_n[display_dims[1]]),
                data,
            )

        msg = "Unknown bit-depth, please submit a bug report" " on Github"
        raise ValueError(msg)

    def get_frame(self, z: int = 0, t: int = 0, c: int = 0, m: int = 0) -> Image.Image:
        """
        Gets the specified frame (z, t, c, m) from image.

        Args:
            z (int): z position
            t (int): time point
            c (int): channel
            m (int): mosaic image

        Returns:
            Pillow Image object
        """
        if self.display_dims != (1, 2):
            msg = (
                "Atypical imaging experiment, please use "
                "get_plane() instead of get_frame()"
            )
            raise ValueError(msg)

        if z >= self.nz:
            msg = "Requested Z frame doesn't exist."
            raise ValueError(msg)

        if t >= self.nt:
            msg = "Requested T frame doesn't exist."
            raise ValueError(msg)

        if c >= self.channels:
            msg = "Requested channel doesn't exist."
            raise ValueError(msg)

        if m >= self.n_mosaic:
            msg = "Requested mosaic image doesn't exist."
            raise ValueError(msg)

        return self.get_plane(
            display_dims=(1, 2), c=c, requested_dims={3: z, 4: t, 10: m}
        )

    def get_iter_t(
        self, z: int = 0, c: int = 0, m: int = 0
    ) -> Generator[Image.Image, Any, None]:
        """
        Returns an iterator over time t at position z and channel c.

        Args:
            z (int): z position
            c (int): channel
            m (int): mosaic image

        Returns:
            Iterator of Pillow Image objects
        """
        for t in range(self.nt):
            yield self.get_frame(z=z, t=t, c=c, m=m)

    def get_iter_c(
        self, z: int = 0, t: int = 0, m: int = 0
    ) -> Generator[Image.Image, Any, None]:
        """
        Returns an iterator over the channels at time t and position z.

        Args:
            z (int): z position
            t (int): time point
            m (int): mosaic image

        Returns:
            Iterator of Pillow Image objects
        """
        for c in range(self.channels):
            yield self.get_frame(z=z, t=t, c=c, m=m)

    def get_iter_z(
        self, t: int = 0, c: int = 0, m: int = 0
    ) -> Generator[Image.Image, Any, None]:
        """
        Returns an iterator over the z series of time t and channel c.

        Args:
            t (int): time point
            c (int): channel
            m (int): mosaic image

        Returns:
            Iterator of Pillow Image objects
        """
        for z in range(self.nz):
            yield self.get_frame(z=z, t=t, c=c, m=m)

    def get_iter_m(
        self, z: int = 0, t: int = 0, c: int = 0
    ) -> Generator[Image.Image, Any, None]:
        """
        Returns an iterator over the z series of time t and channel c.

        Args:
            t (int): time point
            c (int): channel
            z (int): z position

        Returns:
            Iterator of Pillow Image objects
        """
        for m in range(self.n_mosaic):
            yield self.get_frame(z=z, t=t, c=c, m=m)


class LifFile:
    """
    Given a path or buffer to a lif file, returns objects containing
    the image and data.

    This is based on the java openmicroscopy bioformats lif reading code
    that is here: https://github.com/openmicroscopy/bioformats/blob/master/components/formats-gpl/src/loci/formats/in/LIFReader.java

    Attributes:
        xml_header (string): The LIF xml header with tons of data
        xml_root (ElementTree): ElementTree XML representation
        offsets (list): Byte positions of the files
        num_images (int): Number of images
        image_list (dict): Has the keys: path, folder_name, folder_uuid,
            name, image_id, frames

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

    def _recursive_memblock_is_image(
        self, tree: ET.Element, return_list: list[bool] | None = None
    ) -> list[bool]:
        """Creates list of TRUE or FALSE if memblock is image"""

        if return_list is None:
            return_list = []

        children = tree.findall("./Children/Element")
        if len(children) < 1:  # Fix for 'first round'
            children = tree.findall("./Element")
        for item in children:
            has_sub_children = len(item.findall("./Children/Element/Data")) > 0
            is_image = len(item.findall("./Data/Image")) > 0

            # Check to see if the Memblock idnetified in the XML actually has a size,
            # otherwise it won't have an offset
            memory_element = item.find("./Memory")
            if memory_element is not None and int(memory_element.get("Size", 0)) > 0:
                return_list.append(is_image)

            if has_sub_children:
                self._recursive_memblock_is_image(item, return_list)

        return return_list

    def _recursive_image_find(
        self,
        tree: ET.Element,
        return_list: list[dict[str, Any]] | None = None,
        path: str = "",
    ) -> list[dict[str, Any]]:
        """Creates list of images by parsing the XML header recursively"""

        if return_list is None:
            return_list = []

        children = tree.findall("./Children/Element")
        if len(children) < 1:  # Fix for 'first round'
            children = tree.findall("./Element")
        for item in children:
            folder_name = item.attrib["Name"]
            # Grab the .lif filename name on the first execution
            appended_path = folder_name if path == "" else path + "/" + folder_name
            # This finds empty folders
            has_sub_children = len(item.findall("./Children/Element/Data")) > 0

            is_image = len(item.findall("./Data/Image")) > 0

            if is_image:
                # If additional XML data extraction is needed, add it here.

                # Find the dimensions, get them in order
                dims = item.findall("./Data/Image/ImageDescription/" "Dimensions/")

                # Get first two dims, if that fails, set X, Y
                # Todo: Check a 1-d image
                try:
                    dim1 = int(dims[0].attrib["DimID"])
                    dim2 = int(dims[1].attrib["DimID"])
                except (AttributeError, IndexError):
                    dim1 = 1
                    dim2 = 2

                dims_dict = {
                    int(d.attrib["DimID"]): int(d.attrib["NumberOfElements"])
                    for d in dims
                }

                dims_bytes = {
                    int(d.attrib["DimID"]): int(d.attrib["BytesInc"]) for d in dims
                }

                # Get the scale from each image
                scale_dict: dict[int, float | None] = {}
                for d in dims:
                    # Length is not always present, need a try-except
                    dim_n = int(d.attrib["DimID"])
                    try:
                        len_n = float(d.attrib["Length"])

                        # other conversion factor for times needed
                        # returns scale in frames per second
                        if dim_n == 4:
                            scale_dict[dim_n] = (int(dims_dict[dim_n]) - 1) / float(
                                len_n
                            )
                        # Convert from meters to micrometers
                        else:
                            scale_dict[dim_n] = (int(dims_dict[dim_n]) - 1) / (
                                float(len_n) * 10**6
                            )
                    except (AttributeError, ZeroDivisionError):
                        scale_dict[dim_n] = None

                # This code block is to maintain compatibility with programs
                # written before 0.5.0

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

                # The default value needs to be 1, because even if a dimension
                # is missing, it still has to exist. For example, an image that
                # is an x-scan still has one y-dimension.
                dim_x = dims_dict.get(1, 1)
                dim_y = dims_dict.get(2, 1)
                dim_z = dims_dict.get(3, 1)
                dim_t = dims_dict.get(4, 1)
                dim_m = dims_dict.get(10, 1)

                scale_x = scale_dict.get(1)
                scale_y = scale_dict.get(2)
                scale_z = scale_dict.get(3)
                scale_t = scale_dict.get(4)

                # Determine number of channels
                channel_list = item.findall(
                    "./Data/Image/ImageDescription/Channels/ChannelDescription"
                )
                channels_bytes = [int(c.attrib["BytesInc"]) for c in channel_list]
                n_channels = int(len(channel_list))
                # Iterate over each channel, get the resolution
                bit_depth = tuple([int(c.attrib["Resolution"]) for c in channel_list])

                # Get the position data if the image is tiled
                m_pos_list: list[tuple[int, int, float, float]] = []
                if dim_m > 1:
                    for tile in item.findall("./Data/Image/Attachment/Tile"):
                        FieldX = int(tile.attrib["FieldX"])
                        FieldY = int(tile.attrib["FieldY"])
                        PosX = float(tile.attrib["PosX"])
                        PosY = float(tile.attrib["PosY"])

                        m_pos_list.append((FieldX, FieldY, PosX, PosY))

                settings_list = item.findall(
                    "./Data/Image/Attachment/ATLConfocalSettingDefinition"
                )
                settings = settings_list[0].attrib if len(settings_list) > 0 else {}

                data_dict = {
                    "dims": Dims(dim_x, dim_y, dim_z, dim_t, dim_m),
                    "display_dims": (dim1, dim2),
                    "dims_n": dims_dict,
                    "scale_n": scale_dict,
                    "path": path + "/",
                    "name": "/".join((path + "/" + item.attrib["Name"]).split("/")[1:]),
                    "channels": n_channels,
                    "scale": (scale_x, scale_y, scale_z, scale_t),
                    "bit_depth": bit_depth,
                    "mosaic_position": m_pos_list,
                    "settings": settings,
                    "dims_bytes": dims_bytes,
                    "channels_bytes": channels_bytes,
                    # "metadata_xml": item
                }

                return_list.append(data_dict)

            # An image can have sub_children, it is not mutually exclusive
            if has_sub_children:
                self._recursive_image_find(item, return_list, appended_path)

        return return_list

    def __init__(self, filename: FileDescriptorOrPathOrReader):
        self.filename = filename

        if isinstance(filename, (str, bytes, os.PathLike)):
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

        self.image_list = self._recursive_image_find(self.xml_root)

        # If the image is truncated we need to manually add the offsets because
        # the LIF magic bytes aren't present to guide the location.
        if truncated:
            num_truncated = len(self.image_list) - len(self.offsets)
            for _ in range(num_truncated):
                # In the special case of a truncation,
                # append an offset with length zero.
                # This will be taken care of later when the images are retrieved.
                self.offsets.append((truncation_begin, 0))

        # Fix for new LASX version
        if len(self.offsets) - len(self.image_list) > 0:
            is_image_bool_list = self._recursive_memblock_is_image(self.xml_root)
            if False in is_image_bool_list:
                self.offsets = list(compress(self.offsets, is_image_bool_list))

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
        return LifImage(image_info, offsets, self.filename)

    def get_iter_image(self, img_n: int = 0) -> Generator[LifImage, Any, None]:
        """
        Returns an iterator of LifImage objects in the lif file.

        Args:
            img_n (int): Image to start iteration at

        Returns:
            Iterator of LifImage objects.
        """
        for i in range(img_n, len(self.image_list)):
            yield LifImage(self.image_list[i], self.offsets[i], self.filename)
