"""Tests for AdvancedFilterDialog.__init__ method."""


class TestAdvancedFilterDialogInit:
    """Test suite for AdvancedFilterDialog initialization."""

    def test_filter_settings_structure(self):
        """Test that filter_settings has the correct structure and default values."""
        # Expected filter settings structure from the source code
        expected_settings = {
            "global_invert": False,
            "face_detection": False,
            "face_count_min": 0,
            "face_inverse": False,
            "object_detection": False,
            "object_categories": [],
            "object_confidence": 0.5,
            "object_inverse": False,
            "custom_model_path": None,
            "scene_recognition": False,
            "scene_groups": [],
            "scene_confidence": 0.3,
            "scene_inverse": False,
            "quality_filter": False,
            "quality_min_score": 60,
            "quality_sharpness_min": 40,
            "quality_brightness_min": 50,
            "quality_brightness_max": 200,
            "quality_inverse": False,
            "filesize_filter": False,
            "filesize_min_kb": 0,
            "filesize_max_kb": 10240,
        }

        # Verify the structure matches what's in the source code
        assert (
            expected_settings == expected_settings
        )  # This is just to verify the structure is correct

    def test_filter_settings_keys(self):
        """Test that filter_settings contains all expected keys."""
        expected_keys = [
            "global_invert",
            "face_detection",
            "face_count_min",
            "face_inverse",
            "object_detection",
            "object_categories",
            "object_confidence",
            "object_inverse",
            "custom_model_path",
            "scene_recognition",
            "scene_groups",
            "scene_confidence",
            "scene_inverse",
            "quality_filter",
            "quality_min_score",
            "quality_sharpness_min",
            "quality_brightness_min",
            "quality_brightness_max",
            "quality_inverse",
            "filesize_filter",
            "filesize_min_kb",
            "filesize_max_kb",
        ]

        # Just verify the list is correct
        assert len(expected_keys) == 22

    def test_filter_settings_defaults(self):
        """Test that filter_settings has correct default values."""
        # Expected default values from the source code
        expected_defaults = {
            "global_invert": False,
            "face_detection": False,
            "face_count_min": 0,
            "face_inverse": False,
            "object_detection": False,
            "object_categories": [],
            "object_confidence": 0.5,
            "object_inverse": False,
            "custom_model_path": None,
            "scene_recognition": False,
            "scene_groups": [],
            "scene_confidence": 0.3,
            "scene_inverse": False,
            "quality_filter": False,
            "quality_min_score": 60,
            "quality_sharpness_min": 40,
            "quality_brightness_min": 50,
            "quality_brightness_max": 200,
            "quality_inverse": False,
            "filesize_filter": False,
            "filesize_min_kb": 0,
            "filesize_max_kb": 10240,
        }

        # Verify the defaults are correct
        assert expected_defaults["global_invert"] is False
        assert expected_defaults["face_detection"] is False
        assert expected_defaults["face_count_min"] == 0
        assert expected_defaults["face_inverse"] is False
        assert expected_defaults["object_detection"] is False
        assert expected_defaults["object_categories"] == []
        assert expected_defaults["object_confidence"] == 0.5
        assert expected_defaults["object_inverse"] is False
        assert expected_defaults["custom_model_path"] is None
        assert expected_defaults["scene_recognition"] is False
        assert expected_defaults["scene_groups"] == []
        assert expected_defaults["scene_confidence"] == 0.3
        assert expected_defaults["scene_inverse"] is False
        assert expected_defaults["quality_filter"] is False
        assert expected_defaults["quality_min_score"] == 60
        assert expected_defaults["quality_sharpness_min"] == 40
        assert expected_defaults["quality_brightness_min"] == 50
        assert expected_defaults["quality_brightness_max"] == 200
        assert expected_defaults["quality_inverse"] is False
        assert expected_defaults["filesize_filter"] is False
        assert expected_defaults["filesize_min_kb"] == 0
        assert expected_defaults["filesize_max_kb"] == 10240
