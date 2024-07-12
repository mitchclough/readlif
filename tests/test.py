from __future__ import annotations

import os
import unittest
import xml.etree.ElementTree as ET

import requests
from PIL import Image

from readlif.reader import LifFile
from readlif.utilities import get_xml

# Todo: Test a truncated image

TEST_DIR: str = os.path.dirname(os.path.abspath(__file__))


def downloadPrivateFile(filename: str, pwd: str) -> None:
    dl_url = "https://cdn.nimne.com/readlif/" + str(filename)

    if not os.path.exists(os.path.join(TEST_DIR, "private/")):
        os.makedirs(os.path.join(TEST_DIR, "private/"))

    if not os.path.exists(os.path.join(TEST_DIR, "private", filename)):
        with requests.get(dl_url, stream=True, auth=("readlif", pwd)) as r:
            r.raise_for_status()
            with open(os.path.join(TEST_DIR, "private", filename), "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)


class TestReadMethods(unittest.TestCase):
    def test_image_loading(self) -> None:
        # order = c, z, t
        test_array = [[0, 0, 0], [0, 2, 0], [0, 2, 2], [1, 0, 0]]
        for i in test_array:
            c = i[0]
            z = i[1]
            t = i[2]
            ref = Image.open(os.path.join(TEST_DIR, "tiff", f"c{c}z{z}t{t}.tif"))

            obj = LifFile(os.path.join(TEST_DIR, "xyzt_test.lif")).get_image(0)
            test = obj.get_frame(z=z, t=t, c=c)
            self.assertEqual(test.tobytes(), ref.tobytes())

    def test_image_loading_from_buffer(self) -> None:
        # order = c, z, t
        test_array = [[0, 0, 0], [0, 2, 0], [0, 2, 2], [1, 0, 0]]
        for i in test_array:
            c = i[0]
            z = i[1]
            t = i[2]
            ref = Image.open(os.path.join(TEST_DIR, f"tiff/c{c}z{z}t{t}.tif"))

            with open(os.path.join(TEST_DIR, "xyzt_test.lif"), "rb") as open_f:
                obj = LifFile(open_f).get_image(0)
                test = obj.get_frame(z=z, t=t, c=c)
                self.assertEqual(test.tobytes(), ref.tobytes())

    def test_XML_header(self) -> None:
        etroot = get_xml(os.path.join(TEST_DIR, "xyzt_test.lif"))
        test = ET.tostring(etroot, encoding="unicode")
        self.assertEqual(
            test[:50], '<LMSDataContainerHeader Version="2"><Element Name='
        )

    def test_iterators(self) -> None:
        images = list(LifFile(os.path.join(TEST_DIR, "xyzt_test.lif")).get_iter_image())
        self.assertEqual(len(images), 1)

        obj = LifFile(os.path.join(TEST_DIR, "xyzt_test.lif")).get_image(0)
        self.assertEqual(
            repr(obj),
            "'LifImage object with "
            "dimensions: "
            "Dims(x=1024, y=1024, z=3, t=3, m=1)'",
        )

        c_list = list(obj.get_iter_c())
        self.assertEqual(len(c_list), 2)

        t_list = list(obj.get_iter_t())
        self.assertEqual(len(t_list), 3)

        z_list = list(obj.get_iter_z())
        self.assertEqual(len(z_list), 3)

    def test_not_lif_file(self) -> None:
        with self.assertRaises(ValueError):
            LifFile(os.path.join(TEST_DIR, "tiff/c0z0t0.tif"))

    def test_not_that_many_images(self) -> None:
        obj = LifFile(os.path.join(TEST_DIR, "xyzt_test.lif"))
        self.assertEqual(repr(obj), "'LifFile object with 1 image'")

        with self.assertRaises(ValueError):
            obj.get_image(10)

        image = obj.get_image(0)
        with self.assertRaises(ValueError):
            image.get_frame(z=10, t=0, c=0)

        with self.assertRaises(ValueError):
            image.get_frame(z=0, t=10, c=0)

        with self.assertRaises(ValueError):
            image.get_frame(z=0, t=0, c=10)

        with self.assertRaises(ValueError):
            image.get_frame(z=0, t=0, c=0, m=10)

    def test_scale(self) -> None:
        obj = LifFile(os.path.join(TEST_DIR, "xyzt_test.lif")).get_image(0)
        self.assertAlmostEqual(obj.scale[0], 9.8709062997224)

    def test_depth(self) -> None:
        obj = LifFile(os.path.join(TEST_DIR, "xyzt_test.lif")).get_image(0)
        self.assertEqual(obj.bit_depth[0], 8)

    def test_private_images_16bit(self) -> None:
        # These tests are for images that are not public.
        # These images will be pulled from a protected web address
        # during CI testing.
        pwd = os.environ.get("READLIF_TEST_DL_PASSWD")
        if pwd is not None and pwd != "":
            downloadPrivateFile("16bit.lif", pwd)
            downloadPrivateFile("i1c0z2_16b.tif", pwd)
            # Note - readlif produces little endian files,
            # ImageJ makes big endian files for 16bit by default
            obj = LifFile(os.path.join(TEST_DIR, "private/16bit.lif")).get_image(1)

            self.assertEqual(obj.bit_depth[0], 12)

            ref = Image.open("private/i1c0z2_16b.tif")
            test = obj.get_frame(z=2, c=0)

            self.assertEqual(test.tobytes(), ref.tobytes())
        else:
            msg = "READLIF_TEST_DL_PASSWD environment variable not set"
            raise unittest.SkipTest(msg)

    def test_private_images_mosaic(self) -> None:
        # These tests are for images that are not public.
        # These images will be pulled from a protected web address
        # during CI testing.
        pwd = os.environ.get("READLIF_TEST_DL_PASSWD")
        if pwd is not None and pwd != "":
            downloadPrivateFile("tile_002.lif", pwd)
            downloadPrivateFile("i0c1m2z0.tif", pwd)

            obj = LifFile("private/tile_002.lif").get_image(0)
            self.assertEqual(obj.dims.m, 165)

            m_list = list(obj.get_iter_m())
            self.assertEqual(len(m_list), 165)

            ref = Image.open("private/i0c1m2z0.tif")
            test = obj.get_frame(c=1, m=2)

            self.assertEqual(test.tobytes(), ref.tobytes())
        else:
            msg = "READLIF_TEST_DL_PASSWD environment variable not set"
            raise unittest.SkipTest(msg)

    def test_get_plane_on_normal_img(self) -> None:
        # order = c, z, t
        test_array = [[0, 0, 0], [0, 2, 0], [0, 2, 2], [1, 0, 0]]
        for i in test_array:
            c = i[0]
            z = i[1]
            t = i[2]
            ref = Image.open(os.path.join(TEST_DIR, f"tiff/c{c}z{z}t{t}.tif"))

            obj = LifFile(os.path.join(TEST_DIR, "xyzt_test.lif")).get_image(0)
            # 3: z
            # 4: t
            test = obj.get_plane(c=c, requested_dims={3: z, 4: t})
            self.assertEqual(test.tobytes(), ref.tobytes())

    def test_get_plane_on_xz_img(self) -> None:
        ref = Image.open(os.path.join(TEST_DIR, "tiff", "xz_c0_t0.tif"))
        obj = LifFile(os.path.join(TEST_DIR, "testdata_2channel_xz.lif")).get_image(0)
        test = obj.get_plane(c=0, requested_dims={4: 0})
        self.assertEqual(test.tobytes(), ref.tobytes())

        ref2 = Image.open(os.path.join(TEST_DIR, "tiff", "xz_c1_t8.tif"))
        # 3: z
        # 4: t
        test2 = obj.get_plane(c=1, requested_dims={4: 8})
        self.assertEqual(test2.tobytes(), ref2.tobytes())

    def test_arbitrary_plane_on_xzt_img(self) -> None:
        obj = LifFile(
            os.path.join(TEST_DIR, "LeicaLASX_wavelength-sweep_example.lif")
        ).get_image(0)
        with self.assertRaises(NotImplementedError):
            obj.get_plane(display_dims=(1, 5), c=0, requested_dims={2: 31})

    def test_new_lasx(self) -> None:
        obj = LifFile(os.path.join(TEST_DIR, "new_lasx.lif"))
        self.assertEqual(len(obj.image_list), 1)

    def test_settings(self) -> None:
        obj = LifFile(os.path.join(TEST_DIR, "testdata_2channel_xz.lif")).get_image(0)
        self.assertEqual(obj.settings["ObjectiveNumber"], "11506353")


if __name__ == "__main__":
    unittest.main()
