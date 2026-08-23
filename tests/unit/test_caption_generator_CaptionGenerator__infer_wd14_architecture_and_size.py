"""Tests for CaptionGenerator._infer_wd14_architecture_and_size method."""

from unittest.mock import MagicMock
from caption_generator import CaptionGenerator


class TestCaptionGeneratorInferWD14ArchitectureAndSize:
    """Test suite for _infer_wd14_architecture_and_size method."""

    def test_infer_vit_base_patch16(self):
        """Test inference with ViT base patch16 architecture."""
        caption_gen = CaptionGenerator()
        state_dict = {
            "patch_embed.proj.weight": MagicMock(shape=(3, 16, 16, 16)),
            "pos_embed": MagicMock(shape=(1, 197, 384)),
        }
        result = caption_gen._infer_wd14_architecture_and_size(state_dict, "vit-v2")
        assert result == ("vit_base_patch16", 224)

    def test_infer_vit_base_patch14(self):
        """Test inference with ViT base patch14 architecture."""
        caption_gen = CaptionGenerator()
        state_dict = {
            "patch_embed.proj.weight": MagicMock(shape=(3, 14, 14, 14)),
            "pos_embed": MagicMock(shape=(1, 197, 384)),
        }
        result = caption_gen._infer_wd14_architecture_and_size(state_dict, "vit-v2")
        assert result == ("vit_base_patch14", 196)

    def test_infer_swinv2_base(self):
        """Test inference with SwinV2 base architecture."""
        caption_gen = CaptionGenerator()
        state_dict = {
            "patch_embed.proj.weight": MagicMock(shape=(3, 16, 16, 16)),
            "pos_embed": MagicMock(shape=(1, 197, 384)),
        }
        result = caption_gen._infer_wd14_architecture_and_size(state_dict, "swinv2-v2")
        assert result == ("swinv2_base_window12to16_192to256", 256)

    def test_infer_convnextv2_base(self):
        """Test inference with ConvNeXtV2 base architecture."""
        caption_gen = CaptionGenerator()
        state_dict = {
            "patch_embed.proj.weight": MagicMock(shape=(3, 16, 16, 16)),
            "pos_embed": MagicMock(shape=(1, 197, 384)),
        }
        result = caption_gen._infer_wd14_architecture_and_size(
            state_dict, "convnextv2-v2"
        )
        assert result == ("convnextv2_base", 224)

    def test_infer_moat_base(self):
        """Test inference with MOAT base architecture."""
        caption_gen = CaptionGenerator()
        state_dict = {
            "patch_embed.proj.weight": MagicMock(shape=(3, 16, 16, 16)),
            "pos_embed": MagicMock(shape=(1, 197, 384)),
        }
        result = caption_gen._infer_wd14_architecture_and_size(state_dict, "moat-v2")
        assert result == ("mobilevitv2_150", 256)

    def test_infer_without_pos_embed(self):
        """Test inference without pos_embed in state_dict."""
        caption_gen = CaptionGenerator()
        state_dict = {"patch_embed.proj.weight": MagicMock(shape=(3, 16, 16, 16))}
        result = caption_gen._infer_wd14_architecture_and_size(state_dict, "vit-v2")
        assert result == ("vit_base_patch16", 224)

    def test_infer_without_patch_embed(self):
        """Test inference without patch_embed in state_dict."""
        caption_gen = CaptionGenerator()
        state_dict = {"pos_embed": MagicMock(shape=(1, 197, 384))}
        result = caption_gen._infer_wd14_architecture_and_size(state_dict, "vit-v2")
        assert result == ("vit_base_patch16", 224)
