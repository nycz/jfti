import os
from pathlib import Path
from PIL import Image
import shutil
import subprocess
import tempfile
import unittest

from jfti import jfti


def on_all_images(func):
    def wrapper(self):
        for n, old_img, img_path in self.subtests:
            with self.subTest(i=n, name=img_path.name):
                func(self, old_img, img_path)
    return wrapper


TESTDATA_ROOT = Path('testdata')

INCLUDE_SLOW = os.environ.get('INCLUDE_SLOW') == 'yes'


class TestPNG(unittest.TestCase):

    DATA_ROOT = TESTDATA_ROOT / 'png'

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory(prefix='jfti')
        self.subtests = []
        for n, file in enumerate(sorted(self.DATA_ROOT.iterdir())):
            test_path = Path(shutil.copy(file, self.tmpdir.name))
            self.subtests.append((n, Image.open(str(file)), test_path))

    def assertSameImage(self, img1: Image.Image, img2: Image.Image):
        self.assertEqual(list(img1.getdata()), list(img2.getdata()))

    @unittest.skipUnless(INCLUDE_SLOW, 'skipping slow tests')
    @on_all_images
    def test_same_tags_unchanged_image(self, old_img, img_path):
        tags = set(jfti.read_png_tags(img_path))
        jfti.set_png_tags(img_path, tags)
        new_img = Image.open(str(img_path))
        self.assertSameImage(old_img, new_img)

    @unittest.skipUnless(INCLUDE_SLOW, 'skipping slow tests')
    @on_all_images
    def test_new_tags_unchanged_image(self, old_img, img_path):
        jfti.set_png_tags(img_path, {'tag1', 'tag2', 'tag3'})
        new_img = Image.open(str(img_path))
        self.assertSameImage(old_img, new_img)

    @on_all_images
    def test_change_tags(self, old_img, img_path):
        tags = frozenset({'tag1', 'tag2', 'tag3'})
        jfti.set_png_tags(img_path, set(tags))
        new_tags = frozenset(jfti.read_png_tags(img_path))
        self.assertEqual(tags, new_tags)

    @on_all_images
    def test_remove_tags(self, old_img, img_path):
        jfti.set_png_tags(img_path, set())
        new_tags = frozenset(jfti.read_png_tags(img_path))
        self.assertEqual(frozenset(), new_tags)

    @unittest.skipUnless(INCLUDE_SLOW, 'skipping slow tests')
    @on_all_images
    def test_overflow_tags_unchanged_image(self, old_img, img_path):
        tags = frozenset(str(x) * 10 for x in range(300))
        jfti.set_png_tags(img_path, set(tags))
        new_img = Image.open(str(img_path))
        self.assertSameImage(old_img, new_img)

    @on_all_images
    def test_overflow_tags(self, old_img, img_path):
        tags = frozenset(str(x) * 10 for x in range(300))
        jfti.set_png_tags(img_path, set(tags))
        new_tags = frozenset(jfti.read_png_tags(img_path))
        self.assertEqual(tags, new_tags)

    @on_all_images
    def test_dimensions(self, old_img, img_path):
        dimensions = jfti.png_dimensions(img_path)
        self.assertEqual(old_img.size, dimensions)

    def tearDown(self):
        self.tmpdir.cleanup()


class TestJPEG(unittest.TestCase):

    DATA_ROOT = TESTDATA_ROOT / 'jpeg'

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory(prefix='jfti')
        self.subtests = []
        for n, file in enumerate(sorted(self.DATA_ROOT.iterdir())):
            test_path = Path(shutil.copy(file, self.tmpdir.name))
            self.subtests.append((n, Image.open(str(file)), test_path))

    def assertSameImage(self, img1: Image.Image, img2: Image.Image):
        self.assertEqual(list(img1.getdata()), list(img2.getdata()))

    @unittest.skipUnless(INCLUDE_SLOW, 'skipping slow tests')
    @on_all_images
    def test_same_tags_unchanged_image(self, old_img, img_path):
        tags = set(jfti.read_jpeg_tags(img_path))
        jfti.set_jpeg_tags(img_path, tags)
        new_img = Image.open(str(img_path))
        self.assertSameImage(old_img, new_img)

    @unittest.skipUnless(INCLUDE_SLOW, 'skipping slow tests')
    @on_all_images
    def test_new_tags_unchanged_image(self, old_img, img_path):
        jfti.set_jpeg_tags(img_path, {'tag1', 'tag2', 'tag3'})
        new_img = Image.open(str(img_path))
        self.assertSameImage(old_img, new_img)

    @on_all_images
    def test_change_tags(self, old_img, img_path):
        tags = frozenset({'tag1', 'tag2', 'tag3'})
        jfti.set_jpeg_tags(img_path, set(tags))
        new_tags = frozenset(jfti.read_jpeg_tags(img_path))
        self.assertEqual(tags, new_tags)

    @on_all_images
    def test_remove_tags(self, old_img, img_path):
        jfti.set_jpeg_tags(img_path, set())
        new_tags = frozenset(jfti.read_jpeg_tags(img_path))
        self.assertEqual(frozenset(), new_tags)

    @unittest.skipUnless(INCLUDE_SLOW, 'skipping slow tests')
    @on_all_images
    def test_overflow_tags_unchanged_image(self, old_img, img_path):
        tags = frozenset(str(x) * 10 for x in range(300))
        jfti.set_jpeg_tags(img_path, set(tags))
        new_img = Image.open(str(img_path))
        self.assertSameImage(old_img, new_img)

    @on_all_images
    def test_overflow_tags(self, old_img, img_path):
        tags = frozenset(str(x) * 10 for x in range(300))
        jfti.set_jpeg_tags(img_path, set(tags))
        new_tags = frozenset(jfti.read_jpeg_tags(img_path))
        self.assertEqual(tags, new_tags)

    @on_all_images
    def test_dimensions(self, old_img, img_path):
        dimensions = jfti.jpeg_dimensions(img_path)
        self.assertEqual(old_img.size, dimensions)

    @on_all_images
    def test_remove_tags_check_exif(self, old_img, img_path):
        old_data = subprocess.run(['exiv2', '-p', 'v', str(img_path)],
                                  stdout=subprocess.PIPE).stdout
        jfti.set_jpeg_tags(img_path, set())
        new_data = subprocess.run(['exiv2', '-p', 'v', str(img_path)],
                                  stdout=subprocess.PIPE).stdout
        self.assertEqual(old_data, new_data)

    @on_all_images
    def test_change_tags_check_xmp(self, old_img, img_path):
        tags = frozenset({'tag1', 'tag2', 'tag3'})
        old_data = subprocess.run(['exiv2', '-p', 'a', str(img_path)],
                                  stdout=subprocess.PIPE, encoding='utf-8'
                                  ).stdout.splitlines()
        old_data = [x for x in old_data if not x.startswith('Xmp.dc.subject ')]
        jfti.set_jpeg_tags(img_path, set(tags))
        new_data = subprocess.run(['exiv2', '-p', 'a', str(img_path)],
                                  stdout=subprocess.PIPE, encoding='utf-8'
                                  ).stdout.splitlines()
        new_data = [x for x in new_data if not x.startswith('Xmp.dc.subject ')]
        self.assertEqual(old_data, new_data)

    @on_all_images
    def test_remove_tags_check_xmp(self, old_img, img_path):
        old_data = subprocess.run(['exiv2', '-p', 'a', str(img_path)],
                                  stdout=subprocess.PIPE, encoding='utf-8'
                                  ).stdout.splitlines()
        old_data = [x for x in old_data if not x.startswith('Xmp.dc.subject ')]
        jfti.set_jpeg_tags(img_path, set())
        new_data = subprocess.run(['exiv2', '-p', 'a', str(img_path)],
                                  stdout=subprocess.PIPE, encoding='utf-8'
                                  ).stdout.splitlines()
        new_data = [x for x in new_data if not x.startswith('Xmp.dc.subject ')]
        self.assertEqual(old_data, new_data)

    def tearDown(self):
        self.tmpdir.cleanup()


if __name__ == '__main__':
    unittest.main()
