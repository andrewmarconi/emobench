"""
Text preprocessing utilities for SentiCompare.

Provides text cleaning, normalization, and optional data augmentation
for sentiment analysis tasks.
"""

import logging
import re
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Text preprocessing and cleaning utilities.

    Handles various text cleaning operations like URL removal,
    mention removal, special character handling, etc.

    Examples:
        >>> preprocessor = TextPreprocessor(remove_urls=True)
        >>> cleaned = preprocessor.clean_text("Check out https://example.com!")
        >>> print(cleaned)
        'Check out !'
    """

    def __init__(
        self,
        lowercase: bool = False,
        remove_urls: bool = True,
        remove_mentions: bool = False,
        remove_hashtags: bool = False,
        remove_special_chars: bool = False,
        remove_numbers: bool = False,
        remove_extra_spaces: bool = True,
    ):
        """
        Initialize the preprocessor.

        Args:
            lowercase: Convert text to lowercase
            remove_urls: Remove URLs from text
            remove_mentions: Remove @mentions
            remove_hashtags: Remove #hashtags
            remove_special_chars: Remove special characters
            remove_numbers: Remove numbers
            remove_extra_spaces: Remove extra whitespace
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.remove_special_chars = remove_special_chars
        self.remove_numbers = remove_numbers
        self.remove_extra_spaces = remove_extra_spaces

        # Compile regex patterns
        self.url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        self.mention_pattern = re.compile(r"@\w+")
        self.hashtag_pattern = re.compile(r"#\w+")
        self.special_char_pattern = re.compile(r"[^a-zA-Z0-9\s.,!?;:\-\'\"]")
        self.number_pattern = re.compile(r"\d+")
        self.extra_space_pattern = re.compile(r"\s+")

    def clean_text(self, text: str) -> str:
        """
        Clean a single text string.

        Args:
            text: Input text

        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub("", text)

        # Remove mentions
        if self.remove_mentions:
            text = self.mention_pattern.sub("", text)

        # Remove hashtags
        if self.remove_hashtags:
            text = self.hashtag_pattern.sub("", text)

        # Remove numbers
        if self.remove_numbers:
            text = self.number_pattern.sub("", text)

        # Remove special characters
        if self.remove_special_chars:
            text = self.special_char_pattern.sub("", text)

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Remove extra spaces
        if self.remove_extra_spaces:
            text = self.extra_space_pattern.sub(" ", text).strip()

        return text

    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Clean a batch of texts.

        Args:
            texts: List of input texts

        Returns:
            List[str]: List of cleaned texts
        """
        return [self.clean_text(text) for text in texts]

    def clean_dataset(self, examples: Dict, text_column: str = "text") -> Dict:
        """
        Clean text column in a dataset batch (for HuggingFace datasets).

        Args:
            examples: Batch of examples
            text_column: Name of the text column

        Returns:
            Dict: Examples with cleaned text
        """
        examples[text_column] = self.clean_batch(examples[text_column])
        return examples


class LabelMapper:
    """
    Map labels from one format to another.

    Useful for converting multi-class labels to binary,
    or standardizing label formats across datasets.

    Examples:
        >>> mapper = LabelMapper({1: 0, 2: 0, 3: 0, 4: 1, 5: 1})
        >>> mapper.map_label(4)
        1
    """

    def __init__(self, label_mapping: Dict[int, int]):
        """
        Initialize the label mapper.

        Args:
            label_mapping: Dictionary mapping old labels to new labels
        """
        self.label_mapping = label_mapping

    def map_label(self, label: int) -> int:
        """Map a single label."""
        return self.label_mapping.get(label, label)

    def map_batch(self, labels: List[int]) -> List[int]:
        """Map a batch of labels."""
        return [self.map_label(label) for label in labels]

    def map_dataset(self, examples: Dict, label_column: str = "label") -> Dict:
        """Map labels in a dataset batch."""
        examples[label_column] = self.map_batch(examples[label_column])
        return examples


class DataAugmenter:
    """
    Data augmentation for text classification.

    Provides basic augmentation techniques like synonym replacement,
    random insertion, swap, and deletion.

    Note: This is a simple implementation. For production use,
    consider using libraries like nlpaug or textaugment.

    Examples:
        >>> augmenter = DataAugmenter()
        >>> augmented = augmenter.random_swap("This is a great movie")
    """

    def __init__(
        self,
        aug_probability: float = 0.1,
        num_augmented_per_sample: int = 1,
    ):
        """
        Initialize the augmenter.

        Args:
            aug_probability: Probability of augmenting each word
            num_augmented_per_sample: Number of augmented versions per sample
        """
        self.aug_probability = aug_probability
        self.num_augmented_per_sample = num_augmented_per_sample

    def random_swap(self, text: str, n: int = 1) -> str:
        """
        Randomly swap n pairs of words in the text.

        Args:
            text: Input text
            n: Number of swaps

        Returns:
            str: Augmented text
        """
        import random

        words = text.split()
        if len(words) < 2:
            return text

        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]

        return " ".join(words)

    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """
        Randomly delete words with probability p.

        Args:
            text: Input text
            p: Deletion probability

        Returns:
            str: Augmented text
        """
        import random

        words = text.split()
        if len(words) == 1:
            return text

        new_words = [word for word in words if random.random() > p]

        # If all words deleted, return random word
        if len(new_words) == 0:
            return random.choice(words)

        return " ".join(new_words)

    def random_insertion(self, text: str, n: int = 1) -> str:
        """
        Randomly insert n words from the text.

        Args:
            text: Input text
            n: Number of insertions

        Returns:
            str: Augmented text
        """
        import random

        words = text.split()
        if len(words) == 0:
            return text

        for _ in range(n):
            # Insert random word at random position
            word_to_insert = random.choice(words)
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, word_to_insert)

        return " ".join(words)

    def augment(self, text: str, technique: str = "swap") -> str:
        """
        Augment text using specified technique.

        Args:
            text: Input text
            technique: Augmentation technique
                      ('swap', 'deletion', 'insertion')

        Returns:
            str: Augmented text
        """
        if technique == "swap":
            return self.random_swap(text)
        elif technique == "deletion":
            return self.random_deletion(text, self.aug_probability)
        elif technique == "insertion":
            return self.random_insertion(text)
        else:
            raise ValueError(f"Unknown augmentation technique: {technique}")


def validate_dataset(
    data: pd.DataFrame,
    text_column: str,
    label_column: str,
    num_labels: int = 2,
) -> Dict[str, any]:
    """
    Validate dataset integrity and report statistics.

    Args:
        data: DataFrame to validate
        text_column: Name of text column
        label_column: Name of label column
        num_labels: Expected number of labels

    Returns:
        Dict: Validation report
    """
    report = {
        "num_samples": len(data),
        "num_unique_labels": data[label_column].nunique(),
        "expected_labels": num_labels,
        "label_distribution": data[label_column].value_counts().to_dict(),
        "missing_text": data[text_column].isna().sum(),
        "missing_labels": data[label_column].isna().sum(),
        "empty_text": (data[text_column] == "").sum(),
        "avg_text_length": data[text_column].str.len().mean(),
        "min_text_length": data[text_column].str.len().min(),
        "max_text_length": data[text_column].str.len().max(),
    }

    # Check for issues
    issues = []
    if report["missing_text"] > 0:
        issues.append(f"{report['missing_text']} missing text values")
    if report["missing_labels"] > 0:
        issues.append(f"{report['missing_labels']} missing label values")
    if report["empty_text"] > 0:
        issues.append(f"{report['empty_text']} empty text values")
    if report["num_unique_labels"] != num_labels:
        issues.append(
            f"Expected {num_labels} labels, found {report['num_unique_labels']}"
        )

    report["issues"] = issues
    report["valid"] = len(issues) == 0

    return report


def print_validation_report(report: Dict) -> None:
    """Print a formatted validation report."""
    print("\n" + "=" * 60)
    print("Dataset Validation Report".center(60))
    print("=" * 60)
    print(f"Total Samples:        {report['num_samples']:,}")
    print(f"Unique Labels:        {report['num_unique_labels']}")
    print(f"Expected Labels:      {report['expected_labels']}")
    print(f"\nLabel Distribution:")
    for label, count in sorted(report["label_distribution"].items()):
        percentage = (count / report["num_samples"]) * 100
        print(f"  Label {label}:         {count:,} ({percentage:.1f}%)")

    print(f"\nText Statistics:")
    print(f"  Missing Text:     {report['missing_text']}")
    print(f"  Empty Text:       {report['empty_text']}")
    print(f"  Avg Length:       {report['avg_text_length']:.1f} chars")
    print(f"  Min Length:       {report['min_text_length']} chars")
    print(f"  Max Length:       {report['max_text_length']} chars")

    print(f"\nValidation Status:    {'✓ PASS' if report['valid'] else '✗ FAIL'}")
    if report["issues"]:
        print(f"\nIssues Found:")
        for issue in report["issues"]:
            print(f"  - {issue}")
    print("=" * 60 + "\n")
