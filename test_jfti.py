from pathlib import Path
from PIL import Image
import shutil
import tempfile
import unittest

from jfti import jfti


def on_all_images(func):
    def wrapper(self):
        for n, old_img, img_path in self.subtests:
            with self.subTest(i=n):
                func(self, old_img, img_path)
    return wrapper


TESTDATA_ROOT = Path('testdata')


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

    @on_all_images
    def test_same_tags_unchanged_image(self, old_img, img_path):
        tags = set(jfti.read_png_tags(img_path))
        jfti.set_png_tags(img_path, tags)
        new_img = Image.open(str(img_path))
        self.assertSameImage(old_img, new_img)

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

    @on_all_images
    def test_overflow_tags_unchanged_image(self, old_img, img_path):
        tags = frozenset(str(x) * 10 for x in range(100))
        jfti.set_png_tags(img_path, set(tags))
        new_img = Image.open(str(img_path))
        self.assertSameImage(old_img, new_img)

    @on_all_images
    def test_overflow_tags(self, old_img, img_path):
        tags = frozenset(str(x) * 10 for x in range(100))
        jfti.set_png_tags(img_path, set(tags))
        new_tags = frozenset(jfti.read_png_tags(img_path))
        self.assertEqual(tags, new_tags)

    def tearDown(self):
        self.tmpdir.cleanup()


if __name__ == '__main__':
    unittest.main()
