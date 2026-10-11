"""Compose publication comparison plates from saved figures; no FE or optimization."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path("results/validation"), help="Saved validation run directory")
args = parser.parse_args()
root = args.root.resolve()
try:
    title_font = ImageFont.truetype("DejaVuSans.ttf", 42)
    caption_font = ImageFont.truetype("DejaVuSans.ttf", 30)
except OSError:
    title_font = ImageFont.load_default(size=42)
    caption_font = ImageFont.load_default(size=30)

# Keep the exact-density diagnostic separate from illustrative collar assemblies.
for filename, assembly in [("paper_cases.png", False), ("paper_previews.png", True)]:
    plate = Image.new("RGB", (3000, 2560), "white")
    draw = ImageDraw.Draw(plate)
    headings = ["Boundary conditions", "3-axis", "5-axis"]
    if assembly:
        headings[1:] = ["3-axis assembly preview", "5-axis assembly preview"]
    for column, text in enumerate(headings):
        box = draw.textbbox((0, 0), text, font=title_font)
        draw.text((180 + 930*column + (900-box[2]+box[0])/2, 35), text, font=title_font, fill="#222222")
    for row, (case, label) in enumerate([("mbb", "MBB"), ("mount", "Mount"), ("pedal", "Pedal"), ("link", "Link")]):
        y = 110 + 600*row
        draw.text((15, y+255), label, font=title_font, fill="#222222")
        geometry = "paper.png" if assembly and case != "mbb" else "curvature.png"
        panels = [(case+"_3axis", "boundary.png"), (case+"_3axis", geometry), (case+"_5axis", geometry)]
        for column, (run, figure) in enumerate(panels):
            source = root/run/figure
            if not source.is_file():
                parser.error(f"Missing saved figure: {source}; render this run first.")
            with Image.open(source) as image:
                image = image.convert("RGBA")
            image = image.crop(image.getchannel("A").getbbox())
            background = Image.new("RGBA", image.size, "white")
            background.alpha_composite(image)
            image = background.convert("RGB")
            image.thumbnail((860, 540), Image.Resampling.LANCZOS)
            plate.paste(image, (180 + 930*column + (900-image.width)//2, y + (580-image.height)//2))
    caption = "Analytical absolute Gaussian curvature; white to red: 0 to 0.04. Density geometry; collars omitted."
    if assembly:
        caption = "Mount/pedal/link cosmetic collars; FE results and exported meshes use exact density."
    draw.text((180, 2508), caption, font=caption_font, fill="#333333")
    plate.save(root/filename, dpi=(300, 300))
    print(root/filename)

runs = ["global_scale_coarse", "mbb_5axis", "local_z_scale"]
configs = [json.loads((root/run/"config.json").read_text())["optimization"] for run in runs]
summaries = [json.loads((root/run/"summary.json").read_text()) for run in runs]
shared = ["case", "dims_xyz", "mode", "epochs", "seed", "target_volume_fraction"]
if any(summary[key] != summaries[0][key] for summary in summaries for key in shared):
    parser.error("Feature comparison requires matching case, grid, mode, epochs, seed, and volume target.")
feature = Image.new("RGB", (3000, 1000), "white")
draw = ImageDraw.Draw(feature)
for column, (run, config) in enumerate(zip(runs, configs)):
    scale = config["global_scale"]
    fields = config["field_scales"]
    title = f"All-field scale = {float(scale)}"
    if all(field == fields[0] for field in fields):
        local = "Local scales = (" + ",".join(f"{v:g}" for v in fields[0]) + "), all fields"
    elif all(field[:2] == [1, 1] for field in fields):
        local = "Local z scales = (" + ",".join(str(float(field[2])) for field in fields) + ")"
    else:
        local = "Local scales: " + "; ".join(",".join(f"{v:g}" for v in field) for field in fields)
    for y, text, font in [(35, title, title_font), (100, local, caption_font)]:
        box = draw.textbbox((0, 0), text, font=font)
        draw.text((1000*column + (1000-box[2]+box[0])/2, y), text, font=font, fill="#222222")
    source = root/run/"curvature.png"
    if not source.is_file():
        parser.error(f"Missing saved figure: {source}; render this run first.")
    with Image.open(source) as image:
        image = image.convert("RGBA")
    image = image.crop(image.getchannel("A").getbbox())
    background = Image.new("RGBA", image.size, "white")
    background.alpha_composite(image)
    image = background.convert("RGB")
    image.thumbnail((930, 650), Image.Resampling.LANCZOS)
    feature.paste(image, (1000*column + (1000-image.width)//2, 190 + (690-image.height)//2))
summary = summaries[0]
dims = " x ".join(str(n) for n in summary["dims_xyz"])
mode = summary["mode"].replace("axis", "-axis")
caption = (f'{summary["case"].upper()} {dims}; {mode}; volume target {summary["target_volume_fraction"]:.2f}; '
           f'{summary["epochs"]} epochs; seed {summary["seed"]}; absolute curvature range [0,0.04].')
draw.text((80, 910), caption, font=caption_font, fill="#333333")
draw.text((80, 957), "Coordinate scaling changes representable detail; it does not guarantee a fixed physical feature size.",
          font=caption_font, fill="#333333")
feature.save(root/"feature_scale.png", dpi=(300, 300))
print(root/"feature_scale.png")
