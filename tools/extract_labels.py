import os
import shutil

SRC = r"C:\Users\OMEN\Desktop\MAVR-OOD\datasets\RoadAnomaly\images"
DST = r"C:\Users\OMEN\Desktop\MAVR-OOD\datasets\RoadAnomaly\labels"


os.makedirs(DST, exist_ok=True)

count = 0

for item in os.listdir(SRC):
    if item.endswith(".labels"):
        base_name = item.replace(".labels", "")
        semantic_label = os.path.join(SRC, item, "labels_semantic.png")

        if os.path.exists(semantic_label):
            shutil.copy(
                semantic_label,
                os.path.join(DST, base_name + ".png")
            )
            print(f"✔ Copied: {base_name}.png")
            count += 1

print(f"\n✅ Done. Total labels extracted: {count}")
