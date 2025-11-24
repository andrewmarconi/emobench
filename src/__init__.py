"""
SentiCompare - Multi-LLM Sentiment Analysis Benchmark Framework

A comprehensive framework for fine-tuning, evaluating, and comparing
multiple small language models (SLMs) for sentiment analysis tasks.

This package provides tools for:
- Loading and preprocessing sentiment datasets
- Fine-tuning models with LoRA/QLoRA quantization
- Benchmarking performance and inference speed
- Comparing models across multiple metrics
- Interactive visualization and reporting

Example:
    >>> from senticompare import SentimentDataLoader, SentiCompareTrainer
    >>> loader = SentimentDataLoader("imdb", "distilbert-base-uncased")
    >>> trainer = SentiCompareTrainer("distilbert-base-uncased", "imdb")
"""
