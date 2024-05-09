import argparse
from generate import generate_img_wrapper


def main():
    parser = argparse.ArgumentParser(
        description="Generate CT or MRI images using a CycleGAN."
    )
    parser.add_argument(
        "--target-scan-type",
        choices=["mri", "ct"],
        required=True,
        help="Type of scan to generate",
    )
    parser.add_argument(
        "--input-file", required=True, help="Path to the input image file"
    )
    parser.add_argument(
        "--output-file", required=True, help="Path to save the generated image"
    )

    args = parser.parse_args()

    generate_img_wrapper(args.target_scan_type, args.input_file, args.output_file)


if __name__ == "__main__":
    main()
