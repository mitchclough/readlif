from __future__ import annotations

import xml.etree.ElementTree as ET

from ._helpers import FileDescriptorOrPath, check_magic, check_mem, read_u32


def get_xml(filename: FileDescriptorOrPath) -> ET.Element:
    """
    Given a lif file, returns the root xml element for the file's metadata.

    This is useful for debugging.

    Args:
        filename (FileDescriptorOrPath): what file to open?
    """
    with open(filename, "rb") as f:
        check_magic(f)  # read 4 byte, check for magic bytes
        f.seek(8)
        check_mem(f)  # read 1 byte, check for memory byte

        header_len = read_u32(f)  # length of the xml header
        xml_header = f.read(header_len * 2).decode("utf-16")
        return ET.fromstring(xml_header)


def dump_xml(filename: FileDescriptorOrPath, output_filename: FileDescriptorOrPath):
    """
    Given a lif file, outputs an XML file containing the file's metadata.

    This is useful for debugging.

    Args:
        filename: what file to open?
        output_filename: path to the XML file to be created. will overwrite an existing
            file with the same name
    """
    xml_root = get_xml(filename)
    ET.ElementTree(xml_root).write(output_filename, encoding="utf-16")
