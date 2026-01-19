"""
YOLO file handling utilities for reading/writing annotations and YAML files.

This module handles:
- Saving annotations in YOLO format with class mapping comments
- Loading annotations with class ID translation
- Updating label files when classes are renumbered
- Reading/writing yolo.yaml configuration
"""

import os
import yaml


class YoloFileHandler:
    """Handles YOLO format file operations."""

    # Comment formats:
    # # YOLO_CLASSES:0=name0,1=name1,2=name2
    # # YOLO_SOURCE:filename=img.jpg,width=1920,height=1080,sel_x=0.25,sel_y=0.15,sel_w=0.5,sel_h=0.5
    CLASS_MAPPING_PREFIX = "# YOLO_CLASSES:"
    SOURCE_INFO_PREFIX = "# YOLO_SOURCE:"

    @staticmethod
    def build_class_mapping_comment(labels):
        """Build the class mapping comment string.

        Args:
            labels: List of label dicts with 'class_id' and 'name' keys

        Returns:
            str: Comment string like "YOLO_CLASSES:0=name0,1=name1"
        """
        parts = []
        for label in sorted(labels, key=lambda x: x['class_id']):
            class_id = label['class_id']
            name = label.get('name') or f"class_{class_id}"
            # Escape commas and equals in names
            name = name.replace('=', '_').replace(',', '_')
            parts.append(f"{class_id}={name}")
        return "YOLO_CLASSES:" + ",".join(parts)

    @staticmethod
    def parse_class_mapping_comment(comment_line):
        """Parse a class mapping comment and return dict of class_id -> name.

        Args:
            comment_line: Line starting with # YOLO_CLASSES:

        Returns:
            dict mapping class_id (int) to name (str), or None if invalid
        """
        prefix = YoloFileHandler.CLASS_MAPPING_PREFIX
        if not comment_line.startswith(prefix):
            return None

        try:
            mapping_str = comment_line[len(prefix):].strip()
            if not mapping_str:
                return {}

            result = {}
            for part in mapping_str.split(","):
                if "=" in part:
                    id_str, name = part.split("=", 1)
                    result[int(id_str)] = name
            return result
        except (ValueError, IndexError):
            return None

    @staticmethod
    def build_source_info_comment(source_info):
        """Build the source info comment string.

        Args:
            source_info: dict with keys:
                - filename: original source image filename
                - width: original image width in pixels
                - height: original image height in pixels
                - sel_x, sel_y, sel_w, sel_h: selection box as ratios (0.0-1.0)

        Returns:
            str: Comment string like "YOLO_SOURCE:filename=img.jpg,width=1920,..."
        """
        parts = []
        # Escape commas and equals in filename
        filename = source_info.get('filename', '')
        filename = filename.replace('=', '_').replace(',', '_')
        parts.append(f"filename={filename}")
        parts.append(f"width={source_info.get('width', 0)}")
        parts.append(f"height={source_info.get('height', 0)}")
        parts.append(f"sel_x={source_info.get('sel_x', 0):.6f}")
        parts.append(f"sel_y={source_info.get('sel_y', 0):.6f}")
        parts.append(f"sel_w={source_info.get('sel_w', 0):.6f}")
        parts.append(f"sel_h={source_info.get('sel_h', 0):.6f}")
        return "YOLO_SOURCE:" + ",".join(parts)

    @staticmethod
    def parse_source_info_comment(comment_line):
        """Parse a source info comment and return dict of source metadata.

        Args:
            comment_line: Line starting with # YOLO_SOURCE:

        Returns:
            dict with filename, width, height, sel_x, sel_y, sel_w, sel_h
            or None if invalid
        """
        prefix = YoloFileHandler.SOURCE_INFO_PREFIX
        if not comment_line.startswith(prefix):
            return None

        try:
            mapping_str = comment_line[len(prefix):].strip()
            if not mapping_str:
                return None

            result = {}
            for part in mapping_str.split(","):
                if "=" in part:
                    key, value = part.split("=", 1)
                    if key == 'filename':
                        result[key] = value
                    elif key in ('width', 'height'):
                        result[key] = int(value)
                    elif key in ('sel_x', 'sel_y', 'sel_w', 'sel_h'):
                        result[key] = float(value)

            # Validate required fields
            required = ['filename', 'width', 'height', 'sel_x', 'sel_y', 'sel_w', 'sel_h']
            if all(k in result for k in required):
                return result
            return None
        except (ValueError, IndexError):
            return None

    @staticmethod
    def save_annotations(path, annotations, labels, source_info=None):
        """Save annotations in YOLO format with class mapping header.

        Args:
            path: File path for the .txt file
            annotations: List of annotation dicts with 'class_id' and 'ratios'
            labels: List of all label dicts for the class mapping header
            source_info: Optional dict with original image metadata:
                - filename: original source image filename
                - width, height: original image dimensions
                - sel_x, sel_y, sel_w, sel_h: selection box ratios

        YOLO format: class_id x_center y_center width height
        All values normalized to 0.0-1.0 relative to image size
        """
        with open(path, 'w') as f:
            # Write class mapping header comment
            class_mapping = YoloFileHandler.build_class_mapping_comment(labels)
            f.write(f"# {class_mapping}\n")

            # Write source info comment if provided
            if source_info:
                source_comment = YoloFileHandler.build_source_info_comment(source_info)
                f.write(f"# {source_comment}\n")

            for ann in annotations:
                class_id = ann['class_id']
                ratios = ann['ratios']

                # ratios are relative to selection box (which becomes the image)
                # Convert to YOLO center format
                x, y, w, h = ratios
                x_center = x + w / 2
                y_center = y + h / 2

                # Write YOLO format line
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    @staticmethod
    def load_annotations(path, current_labels, default_colors):
        """Load annotations from a YOLO label file.

        Uses the class mapping comment to translate old class IDs to current
        class IDs based on class names.

        Args:
            path: File path for the .txt file
            current_labels: List of current label dicts with 'class_id', 'name', 'color'
            default_colors: List of QColor objects for unknown classes

        Returns:
            tuple: (annotations, source_info)
                - annotations: List of annotation dicts with 'class_id', 'color', 'name', 'ratios'
                - source_info: dict with original image metadata, or None if not present
        """
        if not os.path.exists(path):
            return [], None

        try:
            with open(path, 'r') as f:
                lines = f.readlines()
        except Exception:
            return [], None

        # Build lookup structures
        all_labels = {l['class_id']: l for l in current_labels}

        # Build name -> current_class_id mapping for translation
        name_to_current_id = {}
        for label in current_labels:
            name = label.get('name') or f"class_{label['class_id']}"
            # Normalize name (same as when saving)
            name = name.replace('=', '_').replace(',', '_')
            name_to_current_id[name] = label['class_id']

        # Parse comments from the file
        file_class_mapping = None
        source_info = None
        old_to_new_id = {}

        for line in lines:
            line = line.strip()
            if line.startswith(YoloFileHandler.CLASS_MAPPING_PREFIX):
                file_class_mapping = YoloFileHandler.parse_class_mapping_comment(line)
            elif line.startswith(YoloFileHandler.SOURCE_INFO_PREFIX):
                source_info = YoloFileHandler.parse_source_info_comment(line)

        # Build translation map if file has class mapping
        if file_class_mapping:
            for old_id, name in file_class_mapping.items():
                if name in name_to_current_id:
                    old_to_new_id[old_id] = name_to_current_id[name]

        # Parse annotation lines
        annotations = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) != 5:
                continue

            try:
                file_class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Convert from YOLO center format to top-left format
                x = x_center - width / 2
                y = y_center - height / 2

                # Translate class ID if we have a mapping
                if file_class_id in old_to_new_id:
                    class_id = old_to_new_id[file_class_id]
                else:
                    class_id = file_class_id

                # Get label info
                if class_id in all_labels:
                    label_info = all_labels[class_id]
                    color = label_info['color']
                    name = label_info.get('name', '')
                else:
                    # Unknown class - use default color
                    color = default_colors[class_id % len(default_colors)]
                    name = f"class_{class_id}"

                annotations.append({
                    'class_id': class_id,
                    'color': color,
                    'name': name,
                    'ratios': (x, y, width, height)
                })

            except (ValueError, IndexError):
                continue

        return annotations, source_info

    @staticmethod
    def update_label_files(labels_dir, id_mapping, all_labels):
        """Update all .txt label files with new class IDs after renumbering.

        Args:
            labels_dir: Directory containing .txt label files
            id_mapping: dict mapping old_class_id -> new_class_id
            all_labels: List of current label dicts for the new header

        Returns:
            Number of files updated
        """
        if not os.path.isdir(labels_dir):
            return 0

        updated_count = 0

        for filename in os.listdir(labels_dir):
            if not filename.endswith(".txt"):
                continue

            filepath = os.path.join(labels_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()

                new_lines = []
                file_modified = False

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # Skip and regenerate comment lines
                    if line.startswith("#"):
                        continue

                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            old_class_id = int(parts[0])
                            if old_class_id in id_mapping:
                                # Renumber this annotation
                                parts[0] = str(id_mapping[old_class_id])
                                file_modified = True
                            new_lines.append(" ".join(parts))
                        except ValueError:
                            new_lines.append(line)
                    else:
                        new_lines.append(line)

                # Rewrite file with new header and updated class IDs
                if file_modified or lines:  # Always update to ensure consistent header
                    with open(filepath, 'w') as f:
                        # Write new class mapping header
                        class_mapping = YoloFileHandler.build_class_mapping_comment(all_labels)
                        f.write(f"# {class_mapping}\n")
                        for new_line in new_lines:
                            f.write(f"{new_line}\n")
                    updated_count += 1

            except Exception:
                continue

        return updated_count

    # Header comment for YAML files created by this app
    YAML_HEADER = "# Visual Image Master 2026 - YOLO Dataset Configuration\n# Generated by YOLO Labeling Tool\n\n"

    @staticmethod
    def save_yaml(output_path, labels):
        """Create or update yolo.yaml with class names and colors.

        Args:
            output_path: The YOLO output directory
            labels: List of label dicts with 'class_id', 'name', 'color'
        """
        yaml_path = os.path.join(output_path, "yolo.yaml")

        # Build names dict - ensure names are always non-empty strings
        names = {}
        colors = {}
        for label in labels:
            class_id = label['class_id']
            name = label.get('name', '')
            # Ensure name is a non-empty string (fallback to class_N if empty)
            if not name or not isinstance(name, str):
                name = f"class_{class_id}"
            names[class_id] = name

            # Store color as hex string
            color = label.get('color')
            if color:
                if hasattr(color, 'name'):
                    colors[class_id] = color.name()
                elif isinstance(color, str):
                    colors[class_id] = color

        # Create YAML structure
        yaml_data = {
            'path': output_path,
            'train': 'images',
            'val': 'images',
            'names': names,
            'colors': colors
        }

        # Write YAML file with header comment
        with open(yaml_path, 'w') as f:
            f.write(YoloFileHandler.YAML_HEADER)
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def find_yaml_file(output_path):
        """Find a YAML file in the output directory.

        Looks for any .yaml file, preferring 'yolo.yaml' if it exists.

        Args:
            output_path: The YOLO output directory

        Returns:
            Path to the YAML file, or None if not found
        """
        if not output_path or not os.path.isdir(output_path):
            return None

        # First check for yolo.yaml specifically
        yolo_yaml = os.path.join(output_path, "yolo.yaml")
        if os.path.isfile(yolo_yaml):
            return yolo_yaml

        # Otherwise look for any .yaml file
        try:
            for filename in os.listdir(output_path):
                if filename.endswith('.yaml') or filename.endswith('.yml'):
                    return os.path.join(output_path, filename)
        except OSError:
            pass

        return None

    @staticmethod
    def load_yaml(output_path):
        """Load class labels and colors from a YAML file in the directory.

        Searches for any .yaml file in the directory, preferring 'yolo.yaml'.

        Args:
            output_path: The YOLO output directory

        Returns:
            tuple: (labels, yaml_path) where labels is a list of label dicts
                   with 'class_id', 'name', 'color', or (None, None) if not found
        """
        yaml_path = YoloFileHandler.find_yaml_file(output_path)

        if not yaml_path:
            return None, None

        try:
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)

            if not yaml_data or not isinstance(yaml_data, dict):
                return None, yaml_path

            names = yaml_data.get('names', {})
            colors = yaml_data.get('colors', {})

            if not names:
                return None, yaml_path

            # Build labels list from YAML data
            labels = []
            for class_id, name in names.items():
                # Ensure class_id is an integer
                try:
                    class_id = int(class_id)
                except (ValueError, TypeError):
                    continue

                # Ensure name is a string
                if not isinstance(name, str):
                    name = str(name) if name else f"class_{class_id}"

                label_data = {
                    'class_id': class_id,
                    'name': name
                }

                # Get color if available
                color = colors.get(class_id) or colors.get(str(class_id))
                if color:
                    label_data['color'] = color

                labels.append(label_data)

            # Sort by class_id to maintain order
            labels.sort(key=lambda x: x['class_id'])

            return (labels if labels else None), yaml_path

        except Exception:
            return None, yaml_path
