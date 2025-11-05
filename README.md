# Russian-Bulgarian Parallel Corpus Builder

This repository contains tools for creating a parallel Russian-Bulgarian corpus using sentence transformers for automatic sentence alignment.

## Project Information

This project is part of the research conducted under **RSF Grant No. 25-18-00222**.

For more information about our research group, visit: [CONTROL AND RAISING IN THE LANGUAGES OF EURASIA](https://fortvivlan.github.io/controlandraise/index.html)

## Requirements

This project requires Python 3.8 or higher. Dependencies are managed using `pyproject.toml` and include:

- `numpy` - Numerical computing
- `pandas` - Data manipulation and export
- `razdel` - Russian and Bulgarian sentence segmentation
- `sentence-transformers` - Multilingual sentence embeddings
- `scikit-learn` - Similarity calculations
- `tqdm` - Progress bars
- `openpyxl` - Excel file export

### Installation

To install the required dependencies, run:

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install numpy pandas razdel sentence-transformers scikit-learn tqdm openpyxl
```

## Usage

### Directory Structure

Before running the script, you need to create the following directory structure:

```
rusbulgparcor/
├── ru/              # Folder for Russian texts
│   ├── text1.txt
│   ├── text2.txt
│   └── ...
├── bg/              # Folder for Bulgarian texts
│   ├── text1.txt
│   ├── text2.txt
│   └── ...
└── corpus_bilingual_ru_bg.py
```

**Important:** The text files in both `ru/` and `bg/` folders **must have identical filenames** (e.g., `text1.txt` in both folders). The script matches files by name to create parallel alignments.

### Running the Script

1. Prepare your text files:
   - Place Russian `.txt` files in the `ru/` folder
   - Place corresponding Bulgarian `.txt` files in the `bg/` folder
   - Ensure matching files have the same names

2. Run the corpus builder:

```bash
python corpus_bilingual_ru_bg.py
```

The script will:
- Automatically detect matching file pairs in `ru/` and `bg/` folders
- Load and process texts from both languages
- Split texts into sentences
- Generate multilingual embeddings using sentence transformers
- Align sentences based on semantic similarity and position constraints
- Export the parallel corpus to Excel format

### Output Files

The script generates two Excel files:

1. **`parallel_corpus_ru_bg.xlsx`** - The main parallel corpus with columns:
   - `RU` - Russian sentences
   - `BG` - Bulgarian sentences
   - `Left Context RU` - Previous Russian sentence
   - `Right Context RU` - Next Russian sentence
   - `Left Context BG` - Previous Bulgarian sentence
   - `Right Context BG` - Next Bulgarian sentence
   - `SOURCE` - Source filename

2. **`corpus_ru_bg_summary.xlsx`** - Statistics summary including:
   - Number of alignments per book
   - Average similarity scores
   - Processing statistics

## Configuration

You can adjust alignment parameters by modifying the `Config` class in `corpus_bilingual_ru_bg.py`:

- `SIMILARITY_THRESHOLD` (default: 0.7) - Minimum cosine similarity for alignment
- `WINDOW_SIZE` (default: 10) - Search window size around expected position
- `MIN_SENTENCE_LENGTH` (default: 10) - Minimum character length for valid sentences
- `MAX_LENGTH_RATIO` (default: 2.0) - Maximum allowed length ratio between aligned sentences

## Algorithm

The alignment algorithm uses:
- **Multilingual sentence transformers** for semantic embeddings (paraphrase-multilingual-MiniLM-L12-v2)
- **Position-based constraints** to maintain document structure
- **Cosine similarity** for measuring sentence similarity
- **Length ratio validation** to filter out unlikely alignments
- **Russian as pivot language** (assuming texts are translated from Russian)

## License

This project is licensed under the GNU License. See `LICENSE` file for details.

## Author

**Aleksandra Baiuk**  
Email: alexandra.m.baiuk@gmail.com

## Citation

If you use this corpus or tool in your research, please cite:

```
RSF Grant No. 25-18-00222
Research Group: https://fortvivlan.github.io/controlandraise/index.html
```
