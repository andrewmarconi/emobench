"""
Unit tests for data loading and preprocessing modules.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from src.data.loader import (
    SentimentDataLoader,
    list_available_datasets,
    get_dataset_info,
)
from src.data.preprocessor import (
    TextPreprocessor,
    LabelMapper,
    DataAugmenter,
    validate_dataset,
)


class TestSentimentDataLoader:
    """Tests for SentimentDataLoader class."""

    def test_list_available_datasets(self):
        """Test listing available datasets."""
        datasets = list_available_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) > 0
        assert "imdb" in datasets
        assert "sst2" in datasets

    def test_get_dataset_info(self):
        """Test getting dataset information."""
        info = get_dataset_info("imdb")
        assert info["name"] == "IMDB Movie Reviews"
        assert info["source"] == "huggingface"
        assert "dataset_id" in info

    def test_get_dataset_info_invalid(self):
        """Test getting info for invalid dataset."""
        with pytest.raises(ValueError):
            get_dataset_info("invalid_dataset")

    def test_loader_initialization(self):
        """Test loader initialization."""
        loader = SentimentDataLoader("imdb", "distilbert-base-uncased")
        assert loader.dataset_name == "imdb"
        assert loader.tokenizer_name == "distilbert-base-uncased"
        assert loader.tokenizer is not None

    def test_loader_invalid_dataset(self):
        """Test loader with invalid dataset name."""
        with pytest.raises(ValueError):
            SentimentDataLoader("invalid_dataset", "distilbert-base-uncased")

    def test_get_dataset_info_method(self):
        """Test get_dataset_info method."""
        loader = SentimentDataLoader("imdb", "distilbert-base-uncased")
        info = loader.get_dataset_info()
        assert info["name"] == "imdb"
        assert info["full_name"] == "IMDB Movie Reviews"
        assert "tokenizer" in info


class TestTextPreprocessor:
    """Tests for TextPreprocessor class."""

    def test_basic_cleaning(self):
        """Test basic text cleaning."""
        preprocessor = TextPreprocessor()
        text = "This  is   a    test"
        cleaned = preprocessor.clean_text(text)
        assert cleaned == "This is a test"

    def test_url_removal(self):
        """Test URL removal."""
        preprocessor = TextPreprocessor(remove_urls=True)
        text = "Check out https://example.com for more info"
        cleaned = preprocessor.clean_text(text)
        assert "https://example.com" not in cleaned

    def test_mention_removal(self):
        """Test @mention removal."""
        preprocessor = TextPreprocessor(remove_mentions=True)
        text = "Hey @user123, what do you think?"
        cleaned = preprocessor.clean_text(text)
        assert "@user123" not in cleaned

    def test_hashtag_removal(self):
        """Test #hashtag removal."""
        preprocessor = TextPreprocessor(remove_hashtags=True)
        text = "This is #awesome #cool"
        cleaned = preprocessor.clean_text(text)
        assert "#awesome" not in cleaned
        assert "#cool" not in cleaned

    def test_lowercase(self):
        """Test lowercase conversion."""
        preprocessor = TextPreprocessor(lowercase=True)
        text = "This Is A Test"
        cleaned = preprocessor.clean_text(text)
        assert cleaned == "this is a test"

    def test_number_removal(self):
        """Test number removal."""
        preprocessor = TextPreprocessor(remove_numbers=True)
        text = "I have 123 apples and 456 oranges"
        cleaned = preprocessor.clean_text(text)
        assert "123" not in cleaned
        assert "456" not in cleaned

    def test_special_char_removal(self):
        """Test special character removal."""
        preprocessor = TextPreprocessor(remove_special_chars=True)
        text = "Hello! How are you? #test @user"
        cleaned = preprocessor.clean_text(text)
        # Should keep basic punctuation
        assert "!" in cleaned or "?" in cleaned or cleaned.replace(" ", "").replace(".", "").isalpha()

    def test_empty_text(self):
        """Test handling of empty text."""
        preprocessor = TextPreprocessor()
        assert preprocessor.clean_text("") == ""
        assert preprocessor.clean_text(None) == ""

    def test_batch_cleaning(self):
        """Test cleaning a batch of texts."""
        preprocessor = TextPreprocessor(remove_urls=True)
        texts = [
            "Check out https://example.com",
            "Another  text  with   spaces",
            "Normal text"
        ]
        cleaned = preprocessor.clean_batch(texts)
        assert len(cleaned) == 3
        assert "https://example.com" not in cleaned[0]
        assert "Another text with spaces" == cleaned[1]


class TestLabelMapper:
    """Tests for LabelMapper class."""

    def test_binary_mapping(self):
        """Test mapping multi-class to binary."""
        # Map 1-5 star ratings to binary
        mapping = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}
        mapper = LabelMapper(mapping)

        assert mapper.map_label(1) == 0
        assert mapper.map_label(3) == 0
        assert mapper.map_label(5) == 1

    def test_batch_mapping(self):
        """Test mapping a batch of labels."""
        mapping = {1: 0, 2: 0, 3: 1, 4: 1}
        mapper = LabelMapper(mapping)

        labels = [1, 2, 3, 4, 1, 3]
        mapped = mapper.map_batch(labels)

        assert mapped == [0, 0, 1, 1, 0, 1]

    def test_unmapped_label(self):
        """Test label not in mapping (should return original)."""
        mapping = {1: 0, 2: 1}
        mapper = LabelMapper(mapping)

        assert mapper.map_label(3) == 3  # Not in mapping


class TestDataAugmenter:
    """Tests for DataAugmenter class."""

    def test_random_swap(self):
        """Test random word swap."""
        augmenter = DataAugmenter()
        text = "This is a test sentence"
        augmented = augmenter.random_swap(text, n=1)

        # Words should be same, but order might be different
        original_words = set(text.split())
        augmented_words = set(augmented.split())
        assert original_words == augmented_words

    def test_random_deletion(self):
        """Test random word deletion."""
        augmenter = DataAugmenter()
        text = "This is a test sentence with many words"
        augmented = augmenter.random_deletion(text, p=0.3)

        # Should have fewer or equal words
        assert len(augmented.split()) <= len(text.split())
        # Should not be empty
        assert len(augmented) > 0

    def test_random_insertion(self):
        """Test random word insertion."""
        augmenter = DataAugmenter()
        text = "This is a test"
        augmented = augmenter.random_insertion(text, n=2)

        # Should have more words
        assert len(augmented.split()) >= len(text.split())

    def test_augment_techniques(self):
        """Test different augmentation techniques."""
        augmenter = DataAugmenter()
        text = "This is a test sentence"

        swap = augmenter.augment(text, technique="swap")
        deletion = augmenter.augment(text, technique="deletion")
        insertion = augmenter.augment(text, technique="insertion")

        # All should return strings
        assert isinstance(swap, str)
        assert isinstance(deletion, str)
        assert isinstance(insertion, str)

    def test_invalid_technique(self):
        """Test invalid augmentation technique."""
        augmenter = DataAugmenter()
        with pytest.raises(ValueError):
            augmenter.augment("test", technique="invalid")


class TestDatasetValidation:
    """Tests for dataset validation functions."""

    def test_validate_dataset_valid(self):
        """Test validation with a valid dataset."""
        df = pd.DataFrame({
            "text": ["text1", "text2", "text3"],
            "label": [0, 1, 0]
        })

        report = validate_dataset(df, "text", "label", num_labels=2)

        assert report["valid"] is True
        assert report["num_samples"] == 3
        assert report["num_unique_labels"] == 2
        assert len(report["issues"]) == 0

    def test_validate_dataset_missing_values(self):
        """Test validation with missing values."""
        df = pd.DataFrame({
            "text": ["text1", None, "text3"],
            "label": [0, 1, None]
        })

        report = validate_dataset(df, "text", "label", num_labels=2)

        assert report["valid"] is False
        assert report["missing_text"] == 1
        assert report["missing_labels"] == 1
        assert len(report["issues"]) > 0

    def test_validate_dataset_empty_text(self):
        """Test validation with empty text."""
        df = pd.DataFrame({
            "text": ["text1", "", "text3"],
            "label": [0, 1, 0]
        })

        report = validate_dataset(df, "text", "label", num_labels=2)

        assert report["empty_text"] == 1
        assert len(report["issues"]) > 0

    def test_validate_dataset_wrong_label_count(self):
        """Test validation with wrong number of labels."""
        df = pd.DataFrame({
            "text": ["text1", "text2", "text3"],
            "label": [0, 1, 2]
        })

        report = validate_dataset(df, "text", "label", num_labels=2)

        assert report["num_unique_labels"] == 3
        assert report["expected_labels"] == 2
        assert len(report["issues"]) > 0

    def test_validate_dataset_statistics(self):
        """Test validation statistics calculation."""
        df = pd.DataFrame({
            "text": ["short", "medium length", "very long text here"],
            "label": [0, 1, 0]
        })

        report = validate_dataset(df, "text", "label", num_labels=2)

        assert "avg_text_length" in report
        assert "min_text_length" in report
        assert "max_text_length" in report
        assert report["min_text_length"] == 5  # "short"
        assert report["max_text_length"] == 19  # "very long text here"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
