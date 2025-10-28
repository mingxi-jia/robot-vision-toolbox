"""Entry point for dataset conversion."""

import argparse
import sys
sys.path.append("./")

from pipeline import RealToRobomimicConverter


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert a real dataset to robomimic format."
    )
    parser.add_argument(
        "--real_dataset_path",
        type=str,
        required=True,
        help="Path to the real dataset."
    )
    parser.add_argument(
        "--output_robomimic_path",
        type=str,
        required=True,
        help="Output path for the robomimic HDF5 file."
    )
    args = parser.parse_args()

    converter = RealToRobomimicConverter(
        real_dataset_path=args.real_dataset_path,
        output_robomimic_path=args.output_robomimic_path
    )
    converter.convert()


if __name__ == "__main__":
    main()
