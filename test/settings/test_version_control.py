import unittest

from egenerator.settings.version_control import is_newer_version


class TestIsNewerVersion(unittest.TestCase):
    """Test function `version_control.is_newer_version`."""

    def test_false_on_same_version(self):

        self.assertFalse(
            is_newer_version(
                version_base="1.0.0",
                version_test="1.0.0",
            )
        )

        self.assertFalse(
            is_newer_version(
                version_base="0.0.0-dev",
                version_test="0.0.0-dev",
            )
        )
        self.assertFalse(
            is_newer_version(
                version_base="0.3.4",
                version_test="0.3.4",
            )
        )

    def test_false_on_older_version(self):

        self.assertFalse(
            is_newer_version(
                version_base="1.0.0",
                version_test="0.2.6",
            )
        )

        self.assertFalse(
            is_newer_version(
                version_base="0.0.0-dev",
                version_test="0.0.0",
            )
        )

        self.assertFalse(
            is_newer_version(
                version_base="3.3.4",
                version_test="3.3.2",
            )
        )

        self.assertFalse(
            is_newer_version(
                version_base="3.3.4",
                version_test="3.3.3-dev",
            )
        )

    def test_true_on_newer_version(self):

        self.assertTrue(
            is_newer_version(
                version_base="1.0.0",
                version_test="1.2.6",
            )
        )

        self.assertTrue(
            is_newer_version(
                version_base="0.0.0",
                version_test="0.0.0-dev",
            )
        )

        self.assertTrue(
            is_newer_version(
                version_base="3.3.2",
                version_test="3.3.4",
            )
        )

        self.assertTrue(
            is_newer_version(
                version_base="2.3.4",
                version_test="2.8.3-dev",
            )
        )


if __name__ == "__main__":
    unittest.main()
