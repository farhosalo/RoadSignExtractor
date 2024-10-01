# Traffic Sign Extraction from Dashcam Videos

This project extracts traffic signs from dashcam videos to generate a dataset using the [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) model. The extracted traffic signs are saved as images for further use.

## Features

- **Traffic Sign Detection and Extraction**: Automatically detects and extracts traffic signs from videos.
- **Dataset Generation**: Creates a dataset of detected traffic signs in a temporary directory.

## Prerequisites

- **Python** version >= 3.11
- All required Python packages are listed in the `requirements.txt` file.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/farhosalo/RoadSignExtractor.git
   cd RoadSignExtractor
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To use the project, run the `main.py` script with the path to your image/video file with the -i/--input option. You can also specify the accelerator device by passing the device name with the -d/--device option:

```bash
python ExtractSignsFromVideo.py --device mps --input <path/to/video.mp4>
```

The extracted traffic signs will be saved in the `Signs/` directory.

## Directory Structure

After running the extraction, the following directory structure will be generated:

```
.
├── RoadSignExtractor/
│   └── RoadSignExtractor.py
├── Signs/
├── Weights/
├── Signs/
│   ├── Sign_00000001.jpg
│   ├── Sign_00000002.jpg
│   └── ...
├── main.py
├── requirements.txt
└── README.md
```

## Contributing

Contributions are welcome! If you find any bugs or have ideas for new features, feel free to open an issue or submit a pull request.

### How to Contribute

1. Fork the repository
2. Create a new branch: `git checkout -b my-feature`
3. Make your changes and commit them: `git commit -m 'Add new feature'`
4. Push the branch: `git push origin my-feature`
5. Open a pull request

## License

This project is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
