"""Tests for selecting the exact model checkpoint to evaluate."""

import os
import tempfile
import unittest

from main import resolve_best_checkpoint, resolve_checkpoint_path


class TestCheckpointSelection(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_checkpoint(self, filename):
        path = os.path.join(self.temp_dir.name, filename)
        with open(path, "wb"):
            pass
        return path

    def test_explicit_checkpoint_is_selected(self):
        expected = self.create_checkpoint("seq2seq_attention_run2_best.pth")
        self.create_checkpoint("seq2seq_attention_run1_best.pth")

        selected = resolve_checkpoint_path(
            "seq2seq_attention",
            self.temp_dir.name,
            explicit_path=expected,
        )

        self.assertEqual(selected, expected)

    def test_ambiguous_checkpoints_require_explicit_selection(self):
        self.create_checkpoint("seq2seq_attention_run1_best.pth")
        self.create_checkpoint("seq2seq_attention_run2_best.pth")

        with self.assertRaisesRegex(ValueError, "nhiều checkpoint"):
            resolve_checkpoint_path("seq2seq_attention", self.temp_dir.name)

    def test_missing_checkpoint_fails_instead_of_using_random_weights(self):
        with self.assertRaisesRegex(FileNotFoundError, "--checkpoint"):
            resolve_checkpoint_path("seq2seq_attention", self.temp_dir.name)

    def test_versioned_run_does_not_fall_back_to_legacy_checkpoint(self):
        version_dir = os.path.join(self.temp_dir.name, "seq2seq_attention_v2")
        os.makedirs(version_dir)
        legacy_checkpoint = self.create_checkpoint("seq2seq_attention_best.pth")

        with self.assertRaisesRegex(FileNotFoundError, "seq2seq_attention_v2"):
            resolve_best_checkpoint(
                checkpoints_root=self.temp_dir.name,
                version_ckpt_dir=version_dir,
                base_model="seq2seq_attention",
            )

        self.assertTrue(os.path.isfile(legacy_checkpoint))


if __name__ == "__main__":
    unittest.main()
