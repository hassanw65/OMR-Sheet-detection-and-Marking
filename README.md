# OMR-Sheet-Detection-and-Marking

This project provides a Python-based solution for detecting and grading Optical Mark Recognition (OMR) sheets using image processing techniques. It leverages OpenCV to process scanned answer sheets, identify marked responses, and compute scores based on a predefined answer key.

## Features

- Automatic detection and perspective transformation of OMR sheets.
- Identification of filled bubbles corresponding to multiple-choice answers.
- Comparison with an answer key to calculate scores.
- Visualization of results with marked correct and incorrect answers.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hassanw65/OMR-Sheet-detection-and-Marking.git
   cd OMR-Sheet-detection-and-Marking
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

   *If `requirements.txt` is not provided:*
   ```bash
   pip install opencv-python numpy
   ```

## Usage

1. **Prepare your OMR sheet images:**
   - Place the scanned images of filled OMR sheets in the `input_imgs` directory.
   - Ensure the images are clear and properly scanned.

2. **Define the answer key:**
   - Edit the `ANSWER_KEY` in `bubble_sheet.py`:
     ```python
     ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}
     ```

3. **Run the script:**
   ```bash
   python bubble_sheet.py
   ```

   - The script processes each image in `input_imgs`.
   - Results are visualized and optionally saved.

## Project Structure

```
OMR-Sheet-detection-and-Marking/
├── input_imgs/           # Input images directory
├── utils/                # Utility functions
├── bubble_sheet.py       # Main processing script
├── README.md             # Documentation
└── requirements.txt      # Python packages list
```

## Contributing

Contributions are welcome! Fork the repo and submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgments

This project was inspired by OMR processing techniques and aims to provide an accessible grading solution.
