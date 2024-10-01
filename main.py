import argparse
import filetype
from enum import Enum
from RoadSignExtractor import RoadSignExtractor


class SupportedFileTypes(Enum):
    UNKNOWN = 0
    VIDEO = 1
    IMAGE = 2


def GetFileType(filename):
    mime = filetype.guess_mime(filename)
    if mime is not None:
        if mime.startswith("image/"):
            return SupportedFileTypes.IMAGE
        elif mime.startswith("video/"):
            return SupportedFileTypes.VIDEO

    return SupportedFileTypes.UNKNOWN


def main():
    parser = argparse.ArgumentParser(
        prog="RoadSignExtractor",
        description="Extracts road signs from an image, video or directory",
    )

    parser.add_argument(
        "-d",
        "--device",
        default="cpu",
        choices=["cuda", "mps", "cpu"],
        help="Selects the accelerating device, default is cpu, cuda for NVIDIA GPU and mps for Apple GPU",
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the input image or video"
    )

    args = parser.parse_args()

    extractor = RoadSignExtractor.RoadSignExtractor(Device=args.device)
    extractor.setMinimumSignSize(24, 24)

    inputFile = args.input
    fileType = GetFileType(inputFile)

    if fileType == SupportedFileTypes.VIDEO:
        extractor.extractFromVideo(inputFile)
    elif fileType == SupportedFileTypes.IMAGE:
        extractor.extractFromImage(inputFile)
    else:
        print("Input file format is not supported")


if __name__ == "__main__":
    main()
