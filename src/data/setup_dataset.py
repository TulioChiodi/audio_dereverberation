from pathlib import Path
import shutil


def copy_files(source_dirpath, target_dirpath) -> None:
    target_dirpath.mkdir(parents=True, exist_ok=True)
    for filepath in source_dirpath.glob("**/*.wav"):
        if filepath.parent.name != "rir":
            shutil.copy(filepath, target_dirpath)


def main():
    dataset_size = "large"
    dataset_dirpath_source = Path("../Dataset/repo/AIRCADE/data/dataset_processed/") / dataset_size
    dataset_dirpath_target = Path("data/raw/") / dataset_size / "wavs" / "noisy"
    copy_files(dataset_dirpath_source, dataset_dirpath_target)
    dataset_dirpath_source = Path("../Dataset/repo/AIRCADE/data/dataset_base/")
    dataset_dirpath_target = Path("data/raw/") / dataset_size / "wavs" / "clean"    
    copy_files(dataset_dirpath_source, dataset_dirpath_target)

if __name__ == "__main__":
    main()