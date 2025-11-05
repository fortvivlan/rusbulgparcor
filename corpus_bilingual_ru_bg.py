#########################
## IMPORTS
######################### 
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json

# For sentence processing
from razdel import sentenize

# For multilingual embeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# For visualization and progress
from tqdm import tqdm

print("Libraries imported successfully!")

#########################
# CLASS DEFINITIONS
#########################
# Configuration
class Config:
    # Folder paths
    BG_FOLDER = "bg"
    RU_FOLDER = "ru"
    
    # Model configuration
    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # Alignment parameters
    SIMILARITY_THRESHOLD = 0.7  # Minimum cosine similarity for alignment
    WINDOW_SIZE = 10  # How many sentences around expected position to check
    MIN_SENTENCE_LENGTH = 10  # Minimum character length for valid sentences
    MAX_LENGTH_RATIO = 2.0  # Maximum allowed length ratio between aligned sentences
    
    # Output configuration
    OUTPUT_FILE = "parallel_corpus_ru_bg.json"
    BATCH_SIZE = 32  # For processing embeddings in batches

class TextProcessor:
    """Handles text loading, cleaning, and sentence splitting for multiple languages."""
    
    def __init__(self, min_sentence_length: int = 10):
        self.min_sentence_length = min_sentence_length
        
    def clean_text(self, text: str) -> str:
        """Clean text while preserving sentence structure."""
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = text.strip()
        return text
    
    def split_sentences(self, text: str, language: str = 'russian') -> List[str]:
        """Split text into sentences using appropriate language settings."""
        sentences = [s.text for s in sentenize(text)]
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) >= self.min_sentence_length and sent:
                # Remove sentence fragments and clean up
                if re.search(r'[.!?]$', sent) or len(sent) > 30:
                    cleaned_sentences.append(sent)
        
        return cleaned_sentences
    
    def load_book(self, filepath: str, language: str) -> Dict:
        """Load and process a single book file with encoding detection."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'cp1251', 'latin1', 'iso-8859-1']
            content = None
            
            for encoding in encodings:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                print(f"Failed to decode {filepath} with any encoding")
                return None
            
            # Extract title from filename
            title = Path(filepath).stem
            
            # Clean and split into sentences
            cleaned_content = self.clean_text(content)
            sentences = self.split_sentences(cleaned_content, language)
            
            return {
                'title': title,
                'language': language,
                'filepath': filepath,
                'sentences': sentences,
                'sentence_count': len(sentences)
            }
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def get_matching_books(self, folders: Dict[str, str]) -> List[Tuple[str, Dict[str, str]]]:
        """Find books that exist in both language folders."""
        # Get all files from each folder
        book_files = {}
        for lang, folder in folders.items():
            if os.path.exists(folder):
                book_files[lang] = set(f for f in os.listdir(folder) if f.endswith('.txt'))
            else:
                print(f"Warning: Folder {folder} not found")
                book_files[lang] = set()
        
        # Find common books
        if not book_files:
            return []
        
        common_books = set.intersection(*book_files.values())
        
        # Create tuples of (book_name, {lang: filepath})
        matching_books = []
        for book in sorted(common_books):
            book_paths = {}
            for lang, folder in folders.items():
                book_paths[lang] = os.path.join(folder, book)
            matching_books.append((book, book_paths))
        
        return matching_books
    
class SentenceAligner:
    """Core algorithm for aligning sentences between Russian and Bulgarian."""
    
    def __init__(self, model, similarity_threshold=0.4, window_size=10, max_length_ratio=2.0):
        self.model = model
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size
        self.max_length_ratio = max_length_ratio
    
    def encode_sentences_batch(self, sentences: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode sentences to embeddings in batches to manage memory."""
        if not sentences:
            return np.array([])
        
        embeddings = []
        for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding sentences"):
            batch = sentences[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def find_best_alignment(self, 
                          source_sentences: List[str], 
                          target_sentences: List[str],
                          source_embeddings: Optional[np.ndarray] = None,
                          target_embeddings: Optional[np.ndarray] = None) -> List[Tuple[int, int, float]]:
        """
        Find best sentence alignments between two languages using position constraints.
        Returns list of (source_idx, target_idx, similarity_score) tuples.
        """
        if not source_sentences or not target_sentences:
            return []
        
        # Encode sentences if embeddings not provided
        if source_embeddings is None:
            source_embeddings = self.encode_sentences_batch(source_sentences)
        if target_embeddings is None:
            target_embeddings = self.encode_sentences_batch(target_sentences)
        
        alignments = []
        source_len = len(source_sentences)
        target_len = len(target_sentences)
        
        # Keep track of already aligned target sentences
        used_targets = set()
        
        for src_idx in tqdm(range(source_len), desc="Aligning sentences"):
            # Calculate expected position in target text (proportional scaling)
            expected_pos = int((src_idx / source_len) * target_len)
            
            # Define search window around expected position
            start_pos = max(0, expected_pos - self.window_size)
            end_pos = min(target_len, expected_pos + self.window_size + 1)
            
            best_similarity = -1
            best_target_idx = -1
            
            # Compare with sentences in the window
            for tgt_idx in range(start_pos, end_pos):
                if tgt_idx in used_targets:
                    continue
                
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    source_embeddings[src_idx:src_idx+1],
                    target_embeddings[tgt_idx:tgt_idx+1]
                )[0, 0]
                
                # Check length ratio constraint
                src_length = len(source_sentences[src_idx])
                tgt_length = len(target_sentences[tgt_idx])
                length_ratio = max(src_length, tgt_length) / max(min(src_length, tgt_length), 1)  # Avoid division by zero
                
                if (similarity > best_similarity and 
                    similarity >= self.similarity_threshold and 
                    length_ratio <= self.max_length_ratio):
                    best_similarity = similarity
                    best_target_idx = tgt_idx
            
            # Record alignment if found
            if best_target_idx != -1:
                alignments.append((src_idx, best_target_idx, best_similarity))
                used_targets.add(best_target_idx)
        
        return alignments
    
    def align_bilingual(self, books: Dict[str, Dict]) -> Dict:
        """
        Align sentences between Russian and Bulgarian.
        Strategy: Use Russian as pivot language since books are translated from Russian.
        """
        languages = ['bg', 'ru']
        sentences = {lang: books[lang]['sentences'] for lang in languages}
        
        print("Encoding sentences for both languages...")
        embeddings = {}
        for lang in languages:
            print(f"Encoding {lang}: {len(sentences[lang])} sentences")
            embeddings[lang] = self.encode_sentences_batch(sentences[lang])
        
        print("Finding sentence alignments...")
        print("Using Russian as pivot language (original source)")
        
        # Align RU->BG (Russian as source, Bulgarian as target)
        ru_bg_alignments = self.find_best_alignment(
            sentences['ru'], sentences['bg'],
            embeddings['ru'], embeddings['bg']
        )
        print(f"Found {len(ru_bg_alignments)} RU-BG alignments")
        
        # Create alignment results
        bilingual_alignments = []
        
        for ru_idx, bg_idx, similarity_score in ru_bg_alignments:
            # Check length ratios
            ru_len = len(sentences['ru'][ru_idx])
            bg_len = len(sentences['bg'][bg_idx])
            
            # Calculate length ratio
            length_ratio = max(ru_len, bg_len) / max(min(ru_len, bg_len), 1)
            
            # Accept alignment if similarity is good and length ratio is reasonable
            if (similarity_score >= self.similarity_threshold and
                length_ratio <= self.max_length_ratio):
                
                # Get context sentences (left and right)
                ru_left_context = sentences['ru'][ru_idx - 1] if ru_idx > 0 else ""
                ru_right_context = sentences['ru'][ru_idx + 1] if ru_idx < len(sentences['ru']) - 1 else ""
                bg_left_context = sentences['bg'][bg_idx - 1] if bg_idx > 0 else ""
                bg_right_context = sentences['bg'][bg_idx + 1] if bg_idx < len(sentences['bg']) - 1 else ""
                
                bilingual_alignments.append({
                    'bg_idx': bg_idx,
                    'ru_idx': ru_idx,
                    'bg_sentence': sentences['bg'][bg_idx],
                    'ru_sentence': sentences['ru'][ru_idx],
                    'left_context_ru': ru_left_context,
                    'right_context_ru': ru_right_context,
                    'left_context_bg': bg_left_context,
                    'right_context_bg': bg_right_context,
                    'similarity_score': similarity_score,
                    'length_ratio': length_ratio
                })
        
        print(f"Found {len(bilingual_alignments)} high-quality bilingual alignments")
        print(f"Quality controls: similarity >= {self.similarity_threshold}, length ratio <= {self.max_length_ratio}x")
        
        return {
            'alignments': bilingual_alignments,
            'stats': {
                'bg_sentences': len(sentences['bg']),
                'ru_sentences': len(sentences['ru']),
                'ru_bg_alignments': len(ru_bg_alignments),
                'bilingual_alignments': len(bilingual_alignments)
            }
        }
    
class CorpusBuilder:
    def __init__(self, processor: TextProcessor, aligner: SentenceAligner):
        self.processor = processor
        self.aligner = aligner
        self.corpus = []
        self.stats = {
            'books_processed': 0,
            'total_alignments': 0,
            'sentences_bg': 0,
            'sentences_ru': 0,
            'avg_alignments_per_book': 0
        }
    
    def process_book_pair(self, bg_path: Path, ru_path: Path) -> List[Dict]:
        """Process a pair of books (BG, RU) and return alignments."""
        print(f"Processing: {bg_path.stem}")
        
        # Load books using the correct processor interface
        bg_book = self.processor.load_book(str(bg_path), 'bg')
        ru_book = self.processor.load_book(str(ru_path), 'ru')
        
        if not (bg_book and ru_book):
            print(f"  ❌ Failed to load one or more books")
            return []
        
        print(f"  BG: {bg_book['sentence_count']} sentences")
        print(f"  RU: {ru_book['sentence_count']} sentences")
        
        # Create the books dictionary for the aligner
        books = {
            'bg': bg_book,
            'ru': ru_book
        }
        
        # Perform alignment using the existing interface
        alignment_result = self.aligner.align_bilingual(books)
        alignments = alignment_result['alignments']
        
        # Add source information and clean text for Excel compatibility
        source_name = bg_path.stem.replace('.txt', '')
        processed_alignments = []
        
        for alignment in alignments:
            processed_alignment = {
                'bg_sentence': self._clean_for_excel(alignment['bg_sentence']),
                'ru_sentence': self._clean_for_excel(alignment['ru_sentence']),
                'left_context_ru': self._clean_for_excel(alignment.get('left_context_ru', '')),
                'right_context_ru': self._clean_for_excel(alignment.get('right_context_ru', '')),
                'left_context_bg': self._clean_for_excel(alignment.get('left_context_bg', '')),
                'right_context_bg': self._clean_for_excel(alignment.get('right_context_bg', '')),
                'similarity_score': alignment.get('similarity_score', 0),
                'length_ratio': alignment.get('length_ratio', 0),
                'source': source_name
            }
            processed_alignments.append(processed_alignment)
        
        print(f"  Alignments: {len(processed_alignments)}")
        print()
        
        return processed_alignments
    
    def _clean_for_excel(self, text: str) -> str:
        """Clean text to remove characters that can't be written to Excel."""
        if not text:
            return ""
        
        # Remove or replace illegal characters for Excel
        # These are control characters that Excel can't handle
        illegal_chars = ['\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07', 
                        '\x08', '\x0B', '\x0C', '\x0E', '\x0F', '\x10', '\x11', '\x12', 
                        '\x13', '\x14', '\x15', '\x16', '\x17', '\x18', '\x19', '\x1A', 
                        '\x1B', '\x1C', '\x1D', '\x1E', '\x1F']
        
        # Remove illegal characters
        for char in illegal_chars:
            text = text.replace(char, '')
        
        # Also remove any other problematic Unicode control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Truncate if too long (Excel has a 32,767 character limit per cell)
        if len(text) > 32767:
            text = text[:32767]
        
        return text.strip()
    
    def build_corpus(self, book_pairs: List[Tuple[str, Dict[str, str]]], max_books: Optional[int] = None) -> None:
        """Build the parallel corpus from a list of book pairs."""
        print("🔄 Building bilingual parallel corpus (RU-BG)...")
        
        # Convert to the expected format
        if max_books:
            book_pairs = book_pairs[:max_books]
            
        print(f"Processing {len(book_pairs)} book pairs...")
        print()
        
        all_alignments = []
        
        for book_name, book_paths in book_pairs:
            # Convert paths to Path objects
            bg_path = Path(book_paths['bg'])
            ru_path = Path(book_paths['ru'])
            
            try:
                alignments = self.process_book_pair(bg_path, ru_path)
                all_alignments.extend(alignments)
                
                # Update stats
                self.stats['books_processed'] += 1
            except Exception as e:
                print(f"❌ Error processing {book_name}: {e}")
                continue
        
        # Store results
        self.corpus = all_alignments
        
        # Calculate final stats
        self.stats['total_alignments'] = len(all_alignments)
        if self.stats['books_processed'] > 0:
            self.stats['avg_alignments_per_book'] = self.stats['total_alignments'] / self.stats['books_processed']
        
        # Count unique sentences by language (approximation from alignments)
        unique_bg = set()
        unique_ru = set()
        
        for alignment in all_alignments:
            unique_bg.add(alignment['bg_sentence'])
            unique_ru.add(alignment['ru_sentence'])
        
        self.stats['sentences_bg'] = len(unique_bg)
        self.stats['sentences_ru'] = len(unique_ru)
        
        print("✅ Corpus building completed!")
        print(f"📊 Total alignments: {self.stats['total_alignments']:,}")
        print(f"📚 Books processed: {self.stats['books_processed']}")
        print(f"📈 Average alignments per book: {self.stats['avg_alignments_per_book']:.1f}")
        print()
    
    def export_corpus(self, filename: str, format: str = "json") -> None:
        """Export the corpus in the specified format."""
        if not self.corpus:
            print("❌ No corpus data to export. Build corpus first.")
            return
        
        if format == "json":
            self._export_json(filename)
        elif format == "csv":
            self._export_csv(filename.replace('.json', '.csv'))
        elif format == "xlsx":
            self._export_xlsx(filename.replace('.json', '.xlsx'))
        elif format == "all":
            self._export_json(filename)
            self._export_csv(filename.replace('.json', '.csv'))
            self._export_xlsx(filename.replace('.json', '.xlsx'))
    
    def _export_json(self, filename: str) -> None:
        """Export corpus to JSON format."""
        output_data = {
            'metadata': {
                'total_alignments': len(self.corpus),
                'books_processed': self.stats['books_processed'],
                'creation_date': pd.Timestamp.now().isoformat()
            },
            'alignments': self.corpus
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Exported {len(self.corpus):,} alignments to {filename}")
    
    def _export_csv(self, filename: str) -> None:
        """Export corpus to CSV format."""
        df = pd.DataFrame(self.corpus)
        df = df[['ru_sentence', 'bg_sentence', 'left_context_ru', 'right_context_ru', 'left_context_bg', 'right_context_bg', 'source']]  # Reorder columns
        df.columns = ['RU', 'BG', 'Left Context RU', 'Right Context RU', 'Left Context BG', 'Right Context BG', 'SOURCE']  # Rename columns as requested
        
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"✅ Exported {len(self.corpus):,} alignments to {filename}")
    
    def _export_xlsx(self, filename: str) -> None:
        """Export corpus to Excel format with the requested column structure."""
        df = pd.DataFrame(self.corpus)
        df = df[['ru_sentence', 'bg_sentence', 'left_context_ru', 'right_context_ru', 'left_context_bg', 'right_context_bg', 'source']]  # Reorder columns
        df.columns = ['RU', 'BG', 'Left Context RU', 'Right Context RU', 'Left Context BG', 'Right Context BG', 'SOURCE']  # Rename columns as requested
        
        # Export to Excel with formatting
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Parallel_Corpus_RU_BG', index=False)
            
            # Get the workbook and worksheet
            worksheet = writer.sheets['Parallel_Corpus_RU_BG']
            
            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                
                # Set a reasonable max width
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"✅ Exported {len(self.corpus):,} alignments to {filename}")
        print("📋 Columns: RU (Russian), BG (Bulgarian), Left Context RU, Right Context RU, Left Context BG, Right Context BG, SOURCE (filename)")
    
    def get_stats(self) -> Dict:
        """Return corpus statistics."""
        return self.stats.copy()
    
    def print_sample(self, n: int = 5) -> None:
        """Print a sample of alignments."""
        if not self.corpus:
            print("❌ No corpus data available.")
            return
        
        print(f"📝 Sample of {min(n, len(self.corpus))} alignments:")
        print()
        
        for i, alignment in enumerate(self.corpus[:n]):
            print(f"Alignment #{i+1} (from {alignment['source']}):")
            print(f"  🇷🇺 RU: {alignment['ru_sentence']}")
            print(f"  🇧🇬 BG: {alignment['bg_sentence']}")
            print(f"  📊 Similarity: {alignment.get('similarity_score', 0):.3f}")
            print(f"  📏 Length ratio: {alignment.get('length_ratio', 0):.1f}x")
            
            # Show context if available
            if alignment.get('left_context_ru') or alignment.get('right_context_ru'):
                print(f"  📖 RU Context:")
                if alignment.get('left_context_ru'):
                    print(f"    ← {alignment['left_context_ru'][:60]}{'...' if len(alignment['left_context_ru']) > 60 else ''}")
                if alignment.get('right_context_ru'):
                    print(f"    → {alignment['right_context_ru'][:60]}{'...' if len(alignment['right_context_ru']) > 60 else ''}")
            
            if alignment.get('left_context_bg') or alignment.get('right_context_bg'):
                print(f"  📖 BG Context:")
                if alignment.get('left_context_bg'):
                    print(f"    ← {alignment['left_context_bg'][:60]}{'...' if len(alignment['left_context_bg']) > 60 else ''}")
                if alignment.get('right_context_bg'):
                    print(f"    → {alignment['right_context_bg'][:60]}{'...' if len(alignment['right_context_bg']) > 60 else ''}")
            
            print()

#########################
# INITIALIZATION 
#########################

# load config
config = Config()
print(f"Configuration loaded. Using model: {config.MODEL_NAME}")
print(f"Window size: {config.WINDOW_SIZE}, Similarity threshold: {config.SIMILARITY_THRESHOLD}")
print(f"Max length ratio: {config.MAX_LENGTH_RATIO} (sentences with >2x length difference will be rejected)")

# Initialize text processor
processor = TextProcessor(config.MIN_SENTENCE_LENGTH)

# Find matching books across both languages
folders = {
    'bg': config.BG_FOLDER,
    'ru': config.RU_FOLDER
}
matching_books = processor.get_matching_books(folders)
print(f"Found {len(matching_books)} books available in both languages:")
for book_name, _ in matching_books[:5]:  # Show first 5
    print(f"  - {book_name}")
if len(matching_books) > 5:
    print(f"  ... and {len(matching_books) - 5} more")

# Load the multilingual sentence transformer model
print("Loading multilingual sentence transformer model...")
try:
    model = SentenceTransformer(config.MODEL_NAME)
    print(f"✓ Model loaded successfully: {config.MODEL_NAME}")
    print(f"Model supports languages: {model.get_sentence_embedding_dimension()} dimensions")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying alternative model...")
    try:
        model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        print("✓ Alternative model loaded successfully")
    except Exception as e2:
        print(f"Failed to load any model: {e2}")
        model = None

# Initialize the sentence aligner
if model is not None:
    aligner = SentenceAligner(
        model=model,
        similarity_threshold=config.SIMILARITY_THRESHOLD,
        window_size=config.WINDOW_SIZE,
        max_length_ratio=config.MAX_LENGTH_RATIO
    )
    print("✓ Sentence aligner initialized")
    print(f"  - Russian as pivot language (original source)")
    print(f"  - Similarity threshold: {config.SIMILARITY_THRESHOLD}")
    print(f"  - Length ratio validation: max {config.MAX_LENGTH_RATIO}x difference")
else:
    print("✗ Cannot initialize aligner without model")

# initialize corpus builder
corpus_builder = CorpusBuilder(processor, aligner)

# Display available books
print(f"📚 Processing all {len(matching_books)} book pairs...")
print("Books to be processed:")
for i, (book_name, _) in enumerate(matching_books, 1):
    print(f"  {i:2d}. {book_name}")

print(f"\n🚀 Starting bilingual corpus build (RU-BG)...")


#########################
# EXECUTION
#########################

corpus_builder.build_corpus(matching_books, max_books=None)

# Display final statistics
final_stats = corpus_builder.get_stats()
if final_stats:
    print(f"\n🎉 Final Bilingual Corpus Statistics:")
    print(f"Books successfully processed: {final_stats['books_processed']}")
    print(f"Total bilingual alignments: {final_stats['total_alignments']:,}")
    print(f"Average alignments per book: {final_stats['avg_alignments_per_book']:.1f}")
    print(f"Unique sentences:")
    print(f"  RU: {final_stats['sentences_ru']:,}")
    print(f"  BG: {final_stats['sentences_bg']:,}")
else:
    print("❌ No statistics available")

# Export to XLSX format with the requested column structure
if corpus_builder.corpus:
    print("\n💾 Exporting bilingual parallel corpus to XLSX...")
    
    # Export to XLSX with columns: RU, BG, SOURCE
    corpus_builder.export_corpus("parallel_corpus_ru_bg.xlsx", format="xlsx")
    
    # Also create a summary Excel file with statistics
    print("\n📊 Creating summary statistics...")
    
    # Create a summary dataframe
    summary_data = []
    book_stats = {}
    
    # Group alignments by source
    for alignment in corpus_builder.corpus:
        source = alignment['source']
        if source not in book_stats:
            book_stats[source] = {'count': 0, 'similarities': []}
        book_stats[source]['count'] += 1
        book_stats[source]['similarities'].append(alignment.get('similarity_score', 0))
    
    for i, (source, stats) in enumerate(sorted(book_stats.items()), 1):
        avg_similarity = np.mean(stats['similarities']) if stats['similarities'] else 0
        
        summary_data.append({
            'Book_Number': i,
            'Source': source,
            'Bilingual_Alignments': stats['count'],
            'Avg_Similarity': round(avg_similarity, 3)
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Export summary to Excel
    with pd.ExcelWriter("corpus_ru_bg_summary.xlsx", engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Book_Statistics', index=False)
        
        # Auto-adjust column widths
        worksheet = writer.sheets['Book_Statistics']
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 30)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print("✅ Summary statistics exported to corpus_ru_bg_summary.xlsx")
    
    print(f"\n🎯 Files created:")
    print(f"  📄 parallel_corpus_ru_bg.xlsx - Complete bilingual parallel corpus (RU-BG)")
    print(f"  📊 corpus_ru_bg_summary.xlsx - Processing statistics by book")
    
    # Show sample alignments
    print(f"\n📝 Sample alignments:")
    corpus_builder.print_sample(3)
    
else:
    print("❌ No corpus data to export")