# amazon_nlp_analyzer.py
"""
NLP Analysis of Amazon Product Reviews using spaCy

Goal:
- Perform Named Entity Recognition (NER) to extract product names and brands
- Analyze sentiment (positive/negative) using a rule-based approach

Usage:
    python amazon_nlp_analyzer.py
"""

import spacy
import re
from collections import Counter, defaultdict
import pandas as pd
from typing import List, Dict, Tuple


def load_spacy_model():
    """
    Load spaCy model. Downloads if not available.
    
    Returns:
        spaCy nlp model
    """
    try:
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model 'en_core_web_sm' loaded successfully")
        return nlp
    except OSError:
        print("❌ spaCy model 'en_core_web_sm' not found")
        print("Please install it using: python -m spacy download en_core_web_sm")
        return None


def create_sample_reviews():
    """
    Create sample Amazon product reviews for analysis.
    
    Returns:
        List of sample review texts
    """
    reviews = [
        "I absolutely love my new iPhone 14 Pro from Apple! The camera quality is amazing and the battery life is excellent. Highly recommended!",
        
        "The Samsung Galaxy S23 is disappointing. The screen is good but the battery drains too quickly. Not worth the money.",
        
        "Amazing experience with the Sony WH-1000XM4 headphones! The noise cancellation is fantastic and the sound quality is superb. Best purchase ever!",
        
        "The Nike Air Max 270 shoes are comfortable but the build quality feels cheap. Expected better from Nike for this price point.",
        
        "Terrible experience with this Dell laptop. It's slow, crashes frequently, and customer service was unhelpful. Would not recommend.",
        
        "The Amazon Echo Dot is a great smart speaker! Alexa responds quickly and the sound is clear for its size. Perfect for my bedroom.",
        
        "Love my new MacBook Pro from Apple! Super fast performance, beautiful display, and excellent build quality. Worth every penny!",
        
        "The Google Pixel 7 camera is incredible! Takes amazing photos in low light. However, the battery life could be better.",
        
        "Microsoft Surface Pro is versatile but overpriced. The keyboard feels flimsy and the pen is sold separately which is annoying.",
        
        "Outstanding quality from Bose QuietComfort headphones! Excellent noise cancellation and very comfortable for long flights. Highly satisfied!"
    ]
    
    return reviews


def extract_entities(nlp, text: str) -> Dict[str, List[str]]:
    """
    Extract named entities from text using spaCy NER.
    
    Args:
        nlp: spaCy nlp model
        text: Input text
        
    Returns:
        Dictionary with entity types and their values
    """
    doc = nlp(text)
    entities = defaultdict(list)
    
    for ent in doc.ents:
        entities[ent.label_].append(ent.text)
    
    return dict(entities)


def extract_products_and_brands(nlp, text: str) -> Tuple[List[str], List[str]]:
    """
    Extract product names and brands from text.
    
    Args:
        nlp: spaCy nlp model
        text: Input text
        
    Returns:
        Tuple of (products, brands)
    """
    doc = nlp(text)
    
    # Known brand patterns
    known_brands = {
        'Apple', 'Samsung', 'Sony', 'Nike', 'Dell', 'Amazon', 'Google', 
        'Microsoft', 'Bose', 'iPhone', 'Galaxy', 'MacBook', 'Surface',
        'Echo', 'Pixel', 'QuietComfort', 'WH-1000XM4', 'Air Max'
    }
    
    products = []
    brands = []
    
    # Extract from named entities
    entities = extract_entities(nlp, text)
    
    # Look for organizations (potential brands)
    if 'ORG' in entities:
        for org in entities['ORG']:
            if any(brand.lower() in org.lower() for brand in known_brands):
                brands.append(org)
    
    # Look for products (combination of patterns and entities)
    if 'PRODUCT' in entities:
        products.extend(entities['PRODUCT'])
    
    # Pattern-based extraction for common product patterns
    product_patterns = [
        r'\b(iPhone\s+\d+\s*\w*)\b',
        r'\b(Galaxy\s+S\d+)\b',
        r'\b(MacBook\s+\w+)\b',
        r'\b(Surface\s+\w+)\b',
        r'\b(Echo\s+\w+)\b',
        r'\b(Pixel\s+\d+)\b',
        r'\b(Air\s+Max\s+\d+)\b',
        r'\b(WH-1000XM\d+)\b',
        r'\b(QuietComfort\s*\d*)\b'
    ]
    
    for pattern in product_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        products.extend(matches)
    
    # Brand extraction from known brands
    for brand in known_brands:
        if brand.lower() in text.lower():
            brands.append(brand)
    
    # Remove duplicates while preserving order
    products = list(dict.fromkeys(products))
    brands = list(dict.fromkeys(brands))
    
    return products, brands


def analyze_sentiment_rule_based(text: str) -> Dict[str, any]:
    """
    Analyze sentiment using rule-based approach.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with sentiment analysis results
    """
    # Positive sentiment words
    positive_words = {
        'love', 'amazing', 'excellent', 'fantastic', 'great', 'superb', 'best',
        'outstanding', 'perfect', 'incredible', 'wonderful', 'awesome',
        'brilliant', 'satisfied', 'happy', 'good', 'beautiful', 'fast',
        'comfortable', 'clear', 'worth', 'recommended', 'highly'
    }
    
    # Negative sentiment words
    negative_words = {
        'hate', 'terrible', 'awful', 'bad', 'disappointing', 'poor', 'worst',
        'horrible', 'useless', 'slow', 'cheap', 'flimsy', 'annoying',
        'unhelpful', 'crashes', 'drains', 'overpriced', 'not worth'
    }
    
    # Intensifiers
    intensifiers = {
        'very', 'extremely', 'really', 'absolutely', 'totally', 'completely',
        'highly', 'super', 'quite', 'so', 'too'
    }
    
    # Convert to lowercase for analysis
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    positive_score = 0
    negative_score = 0
    positive_matches = []
    negative_matches = []
    
    for i, word in enumerate(words):
        # Check for intensifiers before sentiment words
        intensifier_multiplier = 1.5 if i > 0 and words[i-1] in intensifiers else 1.0
        
        if word in positive_words:
            positive_score += intensifier_multiplier
            positive_matches.append(word)
        elif word in negative_words:
            negative_score += intensifier_multiplier
            negative_matches.append(word)
    
    # Handle negations (improved approach)
    negation_words = {'not', 'no', 'never', 'nothing', 'nowhere', 'nobody', "n't", 'dont', "don't"}
    negation_found = False
    for neg_word in negation_words:
        if neg_word in text_lower:
            negation_found = True
            break
    
    # If negation is found, reduce positive sentiment and boost negative slightly
    if negation_found:
        positive_score *= 0.3  # Reduce positive sentiment significantly
        negative_score *= 1.2  # Slightly boost negative sentiment
    
    # Determine overall sentiment
    if positive_score > negative_score:
        sentiment = "Positive"
        confidence = positive_score / (positive_score + negative_score) if (positive_score + negative_score) > 0 else 0
    elif negative_score > positive_score:
        sentiment = "Negative"
        confidence = negative_score / (positive_score + negative_score) if (positive_score + negative_score) > 0 else 0
    else:
        sentiment = "Neutral"
        confidence = 0.5
    
    return {
        'sentiment': sentiment,
        'confidence': confidence,
        'positive_score': positive_score,
        'negative_score': negative_score,
        'positive_words_found': positive_matches,
        'negative_words_found': negative_matches
    }


def analyze_single_review(nlp, review: str, review_id: int) -> Dict:
    """
    Analyze a single review for entities and sentiment.
    
    Args:
        nlp: spaCy nlp model
        review: Review text
        review_id: Review identifier
        
    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*60}")
    print(f"REVIEW #{review_id}")
    print(f"{'='*60}")
    print(f"Text: {review}")
    print(f"{'-'*60}")
    
    # Extract entities
    all_entities = extract_entities(nlp, review)
    products, brands = extract_products_and_brands(nlp, review)
    
    # Analyze sentiment
    sentiment_analysis = analyze_sentiment_rule_based(review)
    
    # Display results
    print(f"🏷️  NAMED ENTITY RECOGNITION:")
    if all_entities:
        for entity_type, entities in all_entities.items():
            print(f"   {entity_type}: {', '.join(set(entities))}")
    else:
        print("   No named entities detected")
    
    print(f"\n🏭 PRODUCTS & BRANDS EXTRACTED:")
    print(f"   Products: {', '.join(products) if products else 'None detected'}")
    print(f"   Brands: {', '.join(brands) if brands else 'None detected'}")
    
    print(f"\n😊 SENTIMENT ANALYSIS:")
    print(f"   Sentiment: {sentiment_analysis['sentiment']}")
    print(f"   Confidence: {sentiment_analysis['confidence']:.2f}")
    print(f"   Positive Score: {sentiment_analysis['positive_score']:.1f}")
    print(f"   Negative Score: {sentiment_analysis['negative_score']:.1f}")
    
    if sentiment_analysis['positive_words_found']:
        print(f"   Positive words: {', '.join(set(sentiment_analysis['positive_words_found']))}")
    if sentiment_analysis['negative_words_found']:
        print(f"   Negative words: {', '.join(set(sentiment_analysis['negative_words_found']))}")
    
    return {
        'review_id': review_id,
        'text': review,
        'entities': all_entities,
        'products': products,
        'brands': brands,
        'sentiment': sentiment_analysis
    }


def generate_summary_statistics(results: List[Dict]):
    """
    Generate summary statistics from analysis results.
    
    Args:
        results: List of analysis results
    """
    print(f"\n{'='*70}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*70}")
    
    # Sentiment distribution
    sentiments = [r['sentiment']['sentiment'] for r in results]
    sentiment_counts = Counter(sentiments)
    
    print(f"📊 SENTIMENT DISTRIBUTION:")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(results)) * 100
        print(f"   {sentiment}: {count} reviews ({percentage:.1f}%)")
    
    # Most mentioned brands
    all_brands = []
    for r in results:
        all_brands.extend(r['brands'])
    brand_counts = Counter(all_brands)
    
    print(f"\n🏭 TOP MENTIONED BRANDS:")
    for brand, count in brand_counts.most_common(5):
        print(f"   {brand}: {count} mentions")
    
    # Most mentioned products
    all_products = []
    for r in results:
        all_products.extend(r['products'])
    product_counts = Counter(all_products)
    
    print(f"\n📱 TOP MENTIONED PRODUCTS:")
    for product, count in product_counts.most_common(5):
        print(f"   {product}: {count} mentions")
    
    # Average sentiment confidence
    avg_confidence = sum(r['sentiment']['confidence'] for r in results) / len(results)
    print(f"\n🎯 AVERAGE SENTIMENT CONFIDENCE: {avg_confidence:.2f}")


def main():
    """
    Main function to run the NLP analysis pipeline.
    """
    print("="*70)
    print("AMAZON PRODUCT REVIEWS - NLP ANALYSIS WITH SPACY")
    print("="*70)
    
    # Load spaCy model
    nlp = load_spacy_model()
    if nlp is None:
        return
    
    # Create sample reviews
    reviews = create_sample_reviews()
    print(f"\n📝 Analyzing {len(reviews)} sample Amazon product reviews...")
    
    # Analyze each review
    results = []
    for i, review in enumerate(reviews, 1):
        result = analyze_single_review(nlp, review, i)
        results.append(result)
    
    # Generate summary statistics
    generate_summary_statistics(results)
    
    print(f"\n{'='*70}")
    print("✅ NLP ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("Key Features Demonstrated:")
    print("• Named Entity Recognition (NER) using spaCy")
    print("• Custom product and brand extraction")
    print("• Rule-based sentiment analysis")
    print("• Comprehensive output with statistics")
    print("="*70)


if __name__ == "__main__":
    main() 