"""
reorganise_aanlib.py
====================
Source dataset: https://github.com/xianming-gu/Havard-Medical-Image-Fusion-Datasets

Converts the xianming-gu GitHub download structure:

    Havard-Medical-Image-Fusion-Datasets/
        CT-MRI/
            CT/   2004.png  2005.png ...
            MRI/  2004.png  2005.png ...
        PET-MRI/
            PET/  2004.png  2005.png ...
            MRI/  2004.png  2005.png ...

Into the structure expected by TGFusion dataset.py:

    aanlib/
        ct_mri/
            subject_2004/  ct.png   mri.png
            subject_2005/  ct.png   mri.png
            ...
        mri_pet/
            subject_2004/  mri.png  pet.png
            subject_2005/  mri.png  pet.png
            ...

Usage
------
    python reorganise_aanlib.py \\
        --src  /path/to/Havard-Medical-Image-Fusion-Datasets \\
        --dst  /path/to/aanlib

    # Dry run (preview without copying):
    python reorganise_aanlib.py --src ... --dst ... --dry_run
"""

import argparse
import shutil
from pathlib import Path


def reorganise(src: Path, dst: Path, dry_run: bool = False):

    tasks = [
        # (src_subdir, modality_a_folder, modality_b_folder,
        #  dst_subdir,  out_name_a,       out_name_b)
        ("CT-MRI",  "CT",  "MRI", "ct_mri",  "ct.png",  "mri.png"),
        ("PET-MRI", "MRI", "PET", "mri_pet", "mri.png", "pet.png"),
    ]

    for src_sub, folder_a, folder_b, dst_sub, name_a, name_b in tasks:
        src_a = src / src_sub / folder_a
        src_b = src / src_sub / folder_b

        if not src_a.exists():
            print(f"  ⚠  Skipping {src_sub}: {src_a} not found")
            continue
        if not src_b.exists():
            print(f"  ⚠  Skipping {src_sub}: {src_b} not found")
            continue

        # Find files that exist in BOTH modalities (matched by filename)
        files_a = {f.name: f for f in sorted(src_a.glob("*.png"))}
        files_b = {f.name: f for f in sorted(src_b.glob("*.png"))}
        common  = sorted(set(files_a) & set(files_b))

        print(f"\n── {src_sub} → aanlib/{dst_sub}")
        print(f"   Found {len(files_a)} {folder_a} images, "
              f"{len(files_b)} {folder_b} images, "
              f"{len(common)} matched pairs")

        for fname in common:
            stem     = Path(fname).stem                    # e.g. "2004"
            subj_dir = dst / dst_sub / f"subject_{stem}"

            if not dry_run:
                subj_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(files_a[fname], subj_dir / name_a)
                shutil.copy2(files_b[fname], subj_dir / name_b)

            print(f"   {'[DRY] ' if dry_run else ''}subject_{stem}/ "
                  f"{name_a}, {name_b}")

        print(f"   ✓ {'Would create' if dry_run else 'Created'} "
              f"{len(common)} subject folders in aanlib/{dst_sub}/")


def main():
    p = argparse.ArgumentParser(description="Reorganise AANLIB download for TGFusion")
    p.add_argument("--src", required=True,
                   help="Path to Havard-Medical-Image-Fusion-Datasets/")
    p.add_argument("--dst", required=True,
                   help="Output path for aanlib/ (will be created)")
    p.add_argument("--dry_run", action="store_true",
                   help="Preview without copying any files")
    args = p.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()

    if not src.exists():
        print(f"ERROR: Source not found: {src}")
        return

    print(f"Source : {src}")
    print(f"Dest   : {dst}")
    print(f"Dry run: {args.dry_run}\n")

    reorganise(src, dst, dry_run=args.dry_run)

    print(f"\n{'Preview complete.' if args.dry_run else 'Done!'}")
    if not args.dry_run:
        print(f"\nYour dataset is ready at: {dst}")
        print("\nUse with TGFusion:")
        print(f"  python scripts/train.py --dataset ct_mri  --data_dir {dst}")
        print(f"  python scripts/train.py --dataset mri_pet --data_dir {dst}")


if __name__ == "__main__":
    main()
