import os
import re
import time
import csv
from collections import Counter, deque
from pathlib import Path
from openai import OpenAI
import PyPDF2
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Callable

class GeneralClassifier:
    """
    A general-purpose academic document classifier that can be configured 
    for any classification task by specifying prompts, categories, and output settings.
    """
    
    def __init__(self, api_key: Optional[str] = None, tpm_limit: int = 500000, model: str = "gpt-4o-mini"):
        """
        Initialize the classifier.
        
        Args:
            api_key: OpenAI API key (uses environment variable if None)
            tpm_limit: Tokens per minute limit for OpenAI API
            model: OpenAI model to use (default: "gpt-4o-mini")
                   Supports: "gpt-5" (experimental API) or GPT-4 variants (chat API)
        """
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.token_window = deque()
        self.TPM_LIMIT = tpm_limit
        self.model = model
        self.is_gpt5 = model.lower().startswith("gpt-5")
        print(f"[INFO] Using OpenAI model: {self.model}")
        print(f"[INFO] API type: {'Experimental (responses)' if self.is_gpt5 else 'Standard (chat completions)'}")
        
    def configure_classification(self, 
                               field_name: str,
                               allowed_categories: List[str],
                               classification_prompt_template: str,
                               output_file_prefix: str,
                               max_workers: int = 3,
                               pdf_processor: Optional[Callable] = None,
                               additional_field_extractors: Optional[Dict[str, str]] = None):
        """
        Configure the classification task.
        
        Args:
            field_name: Name of the field to add to BibTeX entries (e.g., "Object", "Discipline", "ResearchType")
            allowed_categories: List of allowed classification categories
            classification_prompt_template: Template for classification prompt. Use {title}, {pdf_text}, {allowed_str}, {additional_fields}
            output_file_prefix: Prefix for output CSV files (e.g., "object_counts", "discipline_analysis")
            max_workers: Number of parallel workers for processing
            pdf_processor: Custom function to extract text from PDFs (uses default if None)
            additional_field_extractors: Dict mapping field names to regex patterns for extracting additional fields
        """
        self.field_name = field_name
        self.allowed_categories = allowed_categories.copy()
        self.classification_prompt_template = classification_prompt_template
        self.output_file_prefix = output_file_prefix
        self.max_workers = max_workers
        self.current_workers = max_workers  # ← Add this line
        self.pdf_processor = pdf_processor or self._default_pdf_processor
        self.additional_field_extractors = additional_field_extractors or {}
        
    def _default_pdf_processor(self, pdf_path: str) -> str:
        """Default PDF text extraction."""
        try:
            with open(pdf_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                pdf_text = ""
                for page in reader.pages:
                    pdf_text += page.extract_text() or ""
                return pdf_text
        except Exception as e:
            print(f"[WARN] Could not read PDF {pdf_path}: {e}")
            return ""
    
    def _log_api_tokens(self, response):
        """Log tokens from OpenAI API response"""
        now = time.time()
        tokens_used = 0
        
        if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
            tokens_used = response.usage.total_tokens
        
        if tokens_used > 0:
            self.token_window.append((now, tokens_used))
            # Drop entries older than 60 seconds
            while self.token_window and now - self.token_window[0][0] > 60:
                self.token_window.popleft()
        
        return tokens_used

    def _check_tpm_status(self):
        """Display current TPM status"""
        current_tpm = sum(tokens for _, tokens in self.token_window)
        percentage = (current_tpm / self.TPM_LIMIT) * 100 if self.TPM_LIMIT > 0 else 0
        
        print(f"    [TPM] {current_tpm}/{self.TPM_LIMIT} tokens/min ({percentage:.1f}%)")
        
        if percentage >= 90:
            print(f"    ⚠️ WARNING: Very close to TPM limit!")
        elif percentage >= 70:
            print(f"    ⚠️ CAUTION: High TPM usage")

    def _should_pause(self):
        """Check if we should pause to avoid TPM limit"""
        current_tpm = sum(tokens for _, tokens in self.token_window)
        return current_tpm >= (self.TPM_LIMIT * 0.95)
    
    def _get_current_tpm_percentage(self):
        """Get current TPM usage as percentage"""
        current_tpm = sum(tokens for _, tokens in self.token_window)
        return (current_tpm / self.TPM_LIMIT) * 100 if self.TPM_LIMIT > 0 else 0
    
    def _adjust_workers(self, current_workers: int, start_time: float) -> int:
        """
        Dynamically adjust number of workers based on TPM usage and runtime.
        
        Rules:
        - If runtime > 1 min and 1 worker and TPM < 40%: add 1 worker
        - If TPM > 90%: remove 1 worker  
        - If 2+ workers and TPM < X%: add 1 worker
        - Max workers: self.max_workers
        - Min workers: 1
        """
        elapsed_minutes = (time.time() - start_time) / 60
        tpm_percentage = self._get_current_tpm_percentage()
        new_workers = current_workers
        
        # Rule 1: If TPM > 90%, reduce workers (safety first)
        if tpm_percentage > 90 and current_workers > 1:
            new_workers = current_workers - 1
            print(f"    🔻 TPM high ({tpm_percentage:.1f}%), reducing workers: {current_workers} → {new_workers}")
                
        
        # Rule 2
        elif tpm_percentage < (100-100/(current_workers+1)-10):
            new_workers = min(current_workers + 1, self.max_workers)
            print(f"    🔺 Low TPM ({tpm_percentage:.1f}%), scaling up: {current_workers} → {new_workers}")
        
        return new_workers

    def _extract_field(self, entry_lines: List[str], field: str) -> Optional[str]:
        """Extract a field from BibTeX entry lines."""
        text = ''.join(entry_lines)
        m = re.search(
            rf'^\s*{re.escape(field)}\s*=\s*(?:\{{(.*?)\}}|"([^"]*)")\s*,?\s*$',
            text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
        )
        if not m:
            return None
        val = m.group(1) if m.group(1) is not None else m.group(2)
        return val.strip() if val is not None else None

    def _extract_file_path(self, entry_lines: List[str]) -> Optional[str]:
        """Extract file path from BibTeX entry."""
        text = ''.join(entry_lines)
        m = re.search(r'file_path\s*=\s*\{(.*?)\}', text, flags=re.IGNORECASE)
        return m.group(1).strip() if m else None

    def _extract_output_text(self, resp) -> str:
        """Extract text from OpenAI response."""
        # Happy path (documented)
        txt = getattr(resp, "output_text", None)
        if isinstance(txt, str) and txt.strip():
            return txt.strip()

        # Walk the structured output
        out = getattr(resp, "output", None) or []
        pieces = []
        for item in out:
            itype = getattr(item, "type", None)
            if itype == "message":
                for seg in getattr(item, "content", []) or []:
                    if getattr(seg, "type", None) == "output_text":
                        t = getattr(seg, "text", None)
                        if isinstance(t, str) and t:
                            pieces.append(t)
            elif itype == "output_text":
                t = getattr(item, "text", None)
                if isinstance(t, str) and t:
                    pieces.append(t)
        joined = " ".join(pieces).strip()
        return joined

    def _classify_entry_lines(self, entry_lines_and_folder):
        """Classify one entry (for parallel processing)."""
        entry_lines, folder_path = entry_lines_and_folder
        
        # Extract standard fields
        title = self._extract_field(entry_lines, 'title')
        pdf_file_name = self._extract_file_path(entry_lines)
        pdf_text = ""
        
        # Extract additional fields if configured
        additional_fields = {}
        for field_name, pattern in self.additional_field_extractors.items():
            additional_fields[field_name] = self._extract_field(entry_lines, field_name)
        
        # Get PDF content
        if pdf_file_name:
            pdf_path = os.path.join(folder_path, pdf_file_name)
            if os.path.exists(pdf_path):
                pdf_text = self.pdf_processor(pdf_path)
            else:
                print(f"[WARN] PDF not found: {pdf_path}")
        
        if pdf_file_name:
            result = self._classify_content(title, pdf_text, additional_fields)
        else:
            result = "no file path"
        
        return result

    def _classify_content(self, title: str, pdf_text: str, additional_fields: Dict, max_retries: int = 10, backoff: float = 1.5):
        """Classify content using the configured prompt."""
        allowed_str = "; ".join(self.allowed_categories)
        
        # Format additional fields for prompt
        additional_fields_str = ""
        if additional_fields:
            field_parts = []
            for field_name, field_value in additional_fields.items():
                if field_value:
                    field_parts.append(f"- {field_name}: {field_value}")
            if field_parts:
                additional_fields_str = "\n" + "\n".join(field_parts)

        # Create the classification prompt
        prompt = self.classification_prompt_template.format(
            title=title or "(none)",
            pdf_text=pdf_text[:100000] if pdf_text else "(none)",
            allowed_str=allowed_str,
            additional_fields=additional_fields_str
        )

        for attempt in range(max_retries):
            try:
                if self.is_gpt5:
                    # Use experimental API for GPT-5
                    resp = self.client.responses.create(
                        model=self.model,
                        input=prompt
                    )
                else:
                    # Use standard chat API for GPT-4 variants
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=50,
                        temperature=0
                    )
                
                # TPM monitoring
                self._log_api_tokens(resp)
                self._check_tpm_status()
                if self._should_pause():
                    print("    💤 Pausing for TPM...")
                    time.sleep(2)
                
                # Extract response based on API type
                if self.is_gpt5:
                    ans = self._extract_output_text(resp)
                else:
                    ans = (resp.choices[0].message.content or "").strip()
                
                # Add new category if not in the list
                if ans not in self.allowed_categories and ans != "no file path":
                    self.allowed_categories.append(ans)
                    
                return ans
                
            except Exception as e:
                print(f"[WARN] API call failed (attempt {attempt+1}): {e}")
                if any(x in str(e) for x in ("429", "timeout", "temporarily", "500", "502", "503", "504")) and attempt < max_retries - 1:
                    time.sleep((attempt + 1) * backoff)
                    continue
                return "Unknown"
        return "Unknown"

    def process_directories(self, base_directory: str = None):
        """
        Process all directories containing BibTeX files.
        
        Args:
            base_directory: Base directory to process (uses current directory if None)
        """
        if base_directory is None:
            try:
                base_directory = os.path.dirname(__file__)
            except NameError:
                base_directory = os.getcwd()

        # Validation
        if not hasattr(self, 'field_name'):
            raise ValueError("Classification not configured. Call configure_classification() first.")

        all_counts = Counter()
        all_results = []
        start_time = time.time()
        
        # Count total entries for progress tracking
        total_entries = 0
        processed_entries = 0

        print("[INFO] Counting total entries for progress tracking...")
        for folder in os.listdir(base_directory):
            folder_path = os.path.join(base_directory, folder)
            if not os.path.isdir(folder_path):
                continue
            bib_files = [f for f in os.listdir(folder_path) if f.endswith('.bib')]
            for bib_file in bib_files:
                bib_path = os.path.join(folder_path, bib_file)
                with open(bib_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    total_entries += content.count('@')

        print(f"[INFO] Found {total_entries} total entries to process.\n")

        # Process each folder
        for folder in os.listdir(base_directory):
            folder_path = os.path.join(base_directory, folder)
            if not os.path.isdir(folder_path):
                continue
                
            bib_files = [f for f in os.listdir(folder_path) if f.endswith('.bib')]
            folder_counts = Counter()
            folder_results = []
            
            print(f"\n[FOLDER] Processing folder: {folder}")

            for bib_file in bib_files:
                bib_path = os.path.join(folder_path, bib_file)
                print(f"  [FILE] Processing: {bib_file}")
                
                # Parse BibTeX entries
                raw_entries = []
                with open(bib_path, 'r', encoding='utf-8') as f:
                    inside = False
                    brace_depth = 0
                    entry = []
                    for line in f:
                        if not inside:
                            if line.lstrip().startswith('@'):
                                inside = True
                                entry = [line]
                                brace_depth = line.count('{') - line.count('}')
                            continue
                        entry.append(line)
                        brace_depth += line.count('{') - line.count('}')
                        if brace_depth <= 0:
                            raw_entries.append(entry[:])
                            entry = []
                            inside = False

                # Parallel classification with dynamic worker adjustment
                print(f"    [ENTRIES] Found {len(raw_entries)} entries in {bib_file}")
                
                entry_data = [(entry_lines, folder_path) for entry_lines in raw_entries]
                results = [None] * len(raw_entries)
                
                # Dynamic worker scaling variables
                current_workers = self.current_workers
                last_adjustment_time = time.time()
                processed_in_batch = 0
                batch_size = max(10, current_workers * 3)  # Process in batches
                
                # Adaptive check interval based on TPM pressure
                def get_adjustment_interval():
                    tmp_pct = self._get_current_tpm_percentage()
                    if tmp_pct > 85:
                        return 3  # Check every 3 seconds when TPM is very high (emergency response)
                    elif tmp_pct > 70:
                        return 5  # Check every 5 seconds when TPM is high
                    else:
                        return 10  # Normal check every 10 seconds when TPM is comfortable
                
                print(f"    [PARALLEL] Starting with {current_workers} workers (dynamic scaling enabled)...")
                
                # Process in batches to allow worker adjustments
                for batch_start in range(0, len(entry_data), batch_size):
                    batch_end = min(batch_start + batch_size, len(entry_data))
                    batch_data = entry_data[batch_start:batch_end]
                    
                    # Check if we need to adjust workers before starting new batch
                    current_adjustment_interval = get_adjustment_interval()
                    if time.time() - last_adjustment_time > current_adjustment_interval:
                        new_workers = self._adjust_workers(current_workers, start_time)
                        if new_workers != current_workers:
                            current_workers = new_workers
                            batch_size = max(10, current_workers * 3)  # Adjust batch size
                            print(f"    ⚡ Adjustment interval: {current_adjustment_interval}s (TPM-adaptive)")
                        last_adjustment_time = time.time()
                    
                    print(f"    [BATCH] Processing batch {batch_start//batch_size + 1} with {current_workers} workers...")
                    
                    with ThreadPoolExecutor(max_workers=current_workers) as ex:
                        batch_futures = {ex.submit(self._classify_entry_lines, data): batch_start + idx 
                                       for idx, data in enumerate(batch_data)}
                        
                        for fut in as_completed(batch_futures):
                            i = batch_futures[fut]
                            try:
                                results[i] = fut.result()
                                processed_entries += 1
                                processed_in_batch += 1
                                
                                # Show progress
                                remaining = total_entries - processed_entries
                                progress_percent = (processed_entries / total_entries) * 100 if total_entries > 0 else 0
                                elapsed_minutes = (time.time() - start_time) / 60
                                tpm_percentage = self._get_current_tpm_percentage()
                                
                                title = self._extract_field(raw_entries[i], 'title')
                                title_short = (title[:50] + '...') if title and len(title) > 50 else (title or 'No title')
                                
                                print(f"    ✅ [{processed_entries}/{total_entries}] ({progress_percent:.1f}%) {title_short}")
                                print(f"       Workers: {current_workers} | TPM: {tpm_percentage:.1f}% | Elapsed: {elapsed_minutes:.1f}min")
                                
                            except Exception as e:
                                results[i] = "Unknown"
                                print(f"    ❌ Error processing entry {i+1}: {e}")
                
                # Final adjustment for next file
                self.current_workers = current_workers

                # Rebuild entries with new field
                final_entries = []
                for entry_lines, result in zip(raw_entries, results):
                    # Remove existing field first
                    cleaned_entry = [line for line in entry_lines 
                                   if not re.match(rf'\s*{re.escape(self.field_name)}\s*=', line, re.IGNORECASE)]
                    
                    # Add new field
                    if cleaned_entry and cleaned_entry[-1].strip() == '}':
                        if len(cleaned_entry) >= 2 and not cleaned_entry[-2].strip().endswith(','):
                            cleaned_entry[-2] = cleaned_entry[-2].rstrip('\n') + ',\n'
                        new_field = f'\t{self.field_name} = {{{result or "Unknown"}}},\n'
                        cleaned_entry.insert(-1, new_field)
                    final_entries.append(cleaned_entry)
                    
                    # Update counters
                    folder_counts[result] += 1
                    all_counts[result] += 1
                    folder_results.append({
                        'title': self._extract_field(entry_lines, 'title'), 
                        self.field_name.lower(): result
                    })
                    all_results.append({
                        'folder': folder, 
                        'title': self._extract_field(entry_lines, 'title'), 
                        self.field_name.lower(): result
                    })

                # Write back BibTeX file
                with open(bib_path, 'w', encoding='utf-8') as out_f:
                    for entry_lines in final_entries:
                        out_f.write(''.join(entry_lines) + '\n')

            # Write folder CSV
            csv_path = os.path.join(folder_path, f"{self.output_file_prefix}.csv")
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([self.field_name, 'Count'])
                for category, count in folder_counts.items():
                    writer.writerow([category, count])

        # Write summary files
        summary_csv = os.path.join(base_directory, f"all_{self.output_file_prefix}.csv")
        with open(summary_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([self.field_name, 'Count'])
            for category, count in all_counts.items():
                writer.writerow([category, count])

        #results_csv = os.path.join(base_directory, f"all_{self.output_file_prefix}_results.csv")
        #with open(results_csv, 'w', newline='', encoding='utf-8') as csvfile:
            #fieldnames = ['folder', 'title', self.field_name.lower()]
            #writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            #writer.writeheader()
            #for row in all_results:
                #writer.writerow(row)

        # Print final statistics
        print(f"\n[DONE] Classification complete. See per-folder and summary CSV files.")
        print(f"\n=== FINAL STATISTICS ===")
        print(f"Total entries processed: {processed_entries}")
        print(f"Success rate: {((processed_entries - all_counts.get('Unknown', 0)) / processed_entries * 100):.1f}% classified successfully" if processed_entries > 0 else "No entries processed")
        print(f"\n=== {self.field_name.upper()} DISTRIBUTION ===")
        for category, count in sorted(all_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / sum(all_counts.values())) * 100 if all_counts.values() else 0
            print(f"{category}: {count} articles ({percentage:.1f}%)")

        return all_counts, all_results


# Example usage configurations
def create_object_classifier(model: str = "gpt-4o-mini"):
    """Example: Create classifier for structural objects"""
    classifier = GeneralClassifier(model=model)
    
    classifier.configure_classification(
        field_name="Object",
        allowed_categories=[
            "Beam", "Column", "Floor", "Bridge", "Slab", "Wall", "Frame", "Truss", "Building"
        ],
        classification_prompt_template="""
Read the following article title and PDF content.
Extract the MAIN structural object that is analyzed or studied in the article (not just mentioned).
If possible, use one from this list: {allowed_str}
If none fit, create a new category.
Return ONLY one label (the main object analyzed). No explanations.

ARTICLE:
- Title: {title}
- PDF Content: {pdf_text}{additional_fields}
""".strip(),
        output_file_prefix="object_counts",
        max_workers=3
    )
    
    return classifier

def create_discipline_classifier(model: str = "gpt-4o-mini"):
    """Example: Create classifier for research disciplines"""
    classifier = GeneralClassifier(model=model)
    
    classifier.configure_classification(
        field_name="Discipline",
        allowed_categories=[
            "Structural Engineering", "Civil Engineering", "Mechanical Engineering", 
            "Computer Science", "Materials Science", "Architecture"
        ],
        classification_prompt_template="""
Analyze the following article and determine the PRIMARY academic discipline.
Choose from these categories: {allowed_str}
If none fit exactly, create a new appropriate discipline category.
Return ONLY the discipline name. No explanations.

ARTICLE:
- Title: {title}
- PDF Content: {pdf_text}{additional_fields}
""".strip(),
        output_file_prefix="discipline_analysis",
        max_workers=4
    )
    
    return classifier

def create_research_type_classifier(model: str = "gpt-4o-mini"):
    """Example: Create classifier for research types"""
    classifier = GeneralClassifier(model=model)
    
    classifier.configure_classification(
        field_name="ResearchType",
        allowed_categories=[
            "Experimental", "Theoretical", "Computational", "Case Study", 
            "Review", "Survey", "Numerical Simulation"
        ],
        classification_prompt_template="""
Identify the PRIMARY research methodology used in this article.
Choose from: {allowed_str}
If the methodology doesn't fit these categories, create a new appropriate category.
Return ONLY the research type. No explanations.

ARTICLE:
- Title: {title}
- PDF Content: {pdf_text}{additional_fields}
""".strip(),
        output_file_prefix="research_type_analysis",
        max_workers=5
    )
    
    return classifier


if __name__ == "__main__":
    # Example usage - you can now choose between GPT-5 and GPT-4 variants
    # GPT-5: Uses experimental responses API
    # GPT-4: Uses standard chat completions API
    
    # Option 1: Use GPT-5 (experimental API)
    MODEL_TO_USE = "gpt-5"
    
    # Option 2: Use GPT-4 variants (standard API)
    # MODEL_TO_USE = "gpt-4o-mini"    # Cost-effective
    # MODEL_TO_USE = "gpt-4o"         # More capable  
    # MODEL_TO_USE = "gpt-4-turbo"    # Previous generation
    
    print("=== OBJECT CLASSIFICATION ===")
    object_classifier = create_object_classifier(model=MODEL_TO_USE)
    object_classifier.process_directories()
    
    print("\n\n=== DISCIPLINE CLASSIFICATION ===")
    discipline_classifier = create_discipline_classifier(model=MODEL_TO_USE)
    discipline_classifier.process_directories()
    
    print("\n\n=== RESEARCH TYPE CLASSIFICATION ===")
    research_type_classifier = create_research_type_classifier(model=MODEL_TO_USE)
    research_type_classifier.process_directories()