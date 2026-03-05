"""
Creates the splash screen image for Visual Image Master 2026.
"""

from PIL import Image, ImageDraw, ImageFont

# Create image (800x400 pixels)
width, height = 800, 400
img = Image.new("RGB", (width, height))
draw = ImageDraw.Draw(img)

# Gradient background (blue to purple)
for y in range(height):
    ratio = y / height
    r = int(50 + (120 - 50) * ratio)
    g = int(100 + (80 - 100) * ratio)
    b = int(200 + (200 - 200) * ratio)
    draw.rectangle([(0, y), (width, y + 1)], fill=(r, g, b))

# Try to use a nice font, fall back to default
try:
    title_font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48
    )
    subtitle_font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24
    )
    loading_font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20
    )
except Exception:
    title_font = ImageFont.load_default()
    subtitle_font = ImageFont.load_default()
    loading_font = ImageFont.load_default()

# Draw application name
title_text = "Visual Image Master"
title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
title_width = title_bbox[2] - title_bbox[0]
title_x = (width - title_width) // 2
draw.text((title_x, 120), title_text, fill=(255, 255, 255), font=title_font)

# Draw year
year_text = "2026"
year_bbox = draw.textbbox((0, 0), year_text, font=subtitle_font)
year_width = year_bbox[2] - year_bbox[0]
year_x = (width - year_width) // 2
draw.text((year_x, 180), year_text, fill=(200, 200, 200), font=subtitle_font)

# Draw simple icon representation (grid of colored squares)
icon_size = 40
icon_spacing = 10
icon_total_width = icon_size * 3 + icon_spacing * 2
icon_start_x = (width - icon_total_width) // 2
icon_start_y = 40

colors = [
    (255, 100, 100),  # Red
    (100, 255, 100),  # Green
    (100, 100, 255),  # Blue
]

for i, color in enumerate(colors):
    x = icon_start_x + i * (icon_size + icon_spacing)
    draw.rectangle(
        [(x, icon_start_y), (x + icon_size, icon_start_y + icon_size)],
        fill=color,
        outline=(255, 255, 255),
        width=2,
    )

# Draw loading text at bottom
loading_text = "Loading..."
loading_bbox = draw.textbbox((0, 0), loading_text, font=loading_font)
loading_width = loading_bbox[2] - loading_bbox[0]
loading_x = (width - loading_width) // 2
draw.text(
    (loading_x, height - 60), loading_text, fill=(255, 255, 255), font=loading_font
)

# Save image
img.save("splash.png")
print("Splash screen created: splash.png")
