# Harsh Rawte 22101A0047
import nltk
import string
import pandas as pd
import re
import warnings
from collections import Counter
from typing import List, Dict
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer, WordNetLemmatizer

warnings.filterwarnings('ignore')

STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
    'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 
    'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 
    'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 
    'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 'from', 'up', 'about', 
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 
    'also', 'like', 'such', 'so', 'than', 'too', 'very', 'just', 'now', 'then', 'here', 
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
    'most', 'other', 'some', 'only', 'own', 'same'
}

def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    words = text.split()
    return [word for word in words if len(word) > 2 and not word.isdigit()]

try:
    print("Attempting to download NLTK data...\n")
    for resource in [
        'punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4'
    ]:
        nltk.download(resource, quiet=True)
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
    print("NLTK data downloaded successfully!\n")
except:
    print("NLTK data download failed. Using fallback options.\n")
    NLTK_AVAILABLE = False

class AdvancedTextProcessor:
    def __init__(self, language='english'):
        self.language = language
        self.porter = PorterStemmer()
        self.snowball = SnowballStemmer(language)
        self.lancaster = LancasterStemmer()
        self.lemmatizer = WordNetLemmatizer() if NLTK_AVAILABLE else None

        try:
            self.stop_words = set(stopwords.words(language)) if NLTK_AVAILABLE else STOP_WORDS
        except:
            self.stop_words = STOP_WORDS

    def get_wordnet_pos(self, word):
        if NLTK_AVAILABLE:
            tag = pos_tag([word])[0][1][0].upper()
            return {'J': 'a', 'N': 'n', 'V': 'v', 'R': 'r'}.get(tag, 'n')
        return 'n'

    def tokenize_text(self, text: str, remove_punc=True, remove_stop=True, min_len=2) -> List[str]:
        tokens = word_tokenize(text.lower()) if NLTK_AVAILABLE else simple_tokenize(text)

        if remove_punc:
            tokens = [t for t in tokens if t not in string.punctuation]
        if remove_stop:
            tokens = [t for t in tokens if t not in self.stop_words]

        return [t for t in tokens if len(t) >= min_len and not t.isdigit()]

    def stem_and_lemmatize_comparison(self, text: str) -> Dict[str, List[str]]:
        tokens = self.tokenize_text(text)

        porter = [self.porter.stem(w) for w in tokens]
        snowball = [self.snowball.stem(w) for w in tokens]
        lancaster = [self.lancaster.stem(w) for w in tokens]

        lemmas = []
        for token in tokens:
            if self.lemmatizer:
                try:
                    pos = self.get_wordnet_pos(token)
                    lemmas.append(self.lemmatizer.lemmatize(token, pos=pos))
                except:
                    lemmas.append(token)
            else:
                lemmas.append(token)

        return {
            'original_tokens': tokens,
            'porter_stems': porter,
            'snowball_stems': snowball,
            'lancaster_stems': lancaster,
            'lemmas': lemmas
        }

    def detailed_analysis(self, text: str) -> Dict:
        result = self.stem_and_lemmatize_comparison(text)
        df = pd.DataFrame({
            'Original': result['original_tokens'],
            'Porter': result['porter_stems'],
            'Snowball': result['snowball_stems'],
            'Lancaster': result['lancaster_stems'],
            'Lemma': result['lemmas']
        })

        unique_tokens = len(set(result['original_tokens']))

        stats = {
            'total_tokens': len(result['original_tokens']),
            'unique_tokens': unique_tokens,
            'porter_unique': len(set(result['porter_stems'])),
            'snowball_unique': len(set(result['snowball_stems'])),
            'lancaster_unique': len(set(result['lancaster_stems'])),
            'lemma_unique': len(set(result['lemmas'])),
        }

        for key in ['porter', 'snowball', 'lancaster', 'lemma']:
            stats[f'{key}_reduction'] = (
                (unique_tokens - stats[f'{key}_unique']) / unique_tokens * 100
                if unique_tokens > 0 else 0
            )

        return {'dataframe': df, 'statistics': stats, 'raw_results': result}

    def find_differences(self, text: str) -> pd.DataFrame:
        res = self.stem_and_lemmatize_comparison(text)
        diffs = []
        for i, word in enumerate(res['original_tokens']):
            forms = [res['porter_stems'][i], res['snowball_stems'][i],
                     res['lancaster_stems'][i], res['lemmas'][i]]
            if len(set(forms)) > 1:
                diffs.append({
                    'Original': word,
                    'Porter': forms[0],
                    'Snowball': forms[1],
                    'Lancaster': forms[2],
                    'Lemma': forms[3],
                    'Unique_Forms': len(set(forms))
                })
        return pd.DataFrame(diffs)

    def show_comparison_table(self, text: str):
        res = self.stem_and_lemmatize_comparison(text)
        print(f"\n{'Original':<15}{'Porter':<15}{'Snowball':<15}{'Lancaster':<15}{'Lemma':<15}")
        print("-" * 75)
        for i in range(len(res['original_tokens'])):
            print(f"{res['original_tokens'][i]:<15}{res['porter_stems'][i]:<15}"
                  f"{res['snowball_stems'][i]:<15}{res['lancaster_stems'][i]:<15}"
                  f"{res['lemmas'][i]:<15}")

def demonstrate_stemmer_lemmatizer_characteristics():
    processor = AdvancedTextProcessor()
    test_words = [
        'scoring', 'dribbling', 'passes', 'teams', 'players', 'better', 'good',
        'strikers', 'defending', 'played', 'running', 'feet', 'goals', 'won', 'losses'
    ]
    print("=== STEMMER vs LEMMATIZER CHARACTERISTICS ===\n")
    print(f"{'Word':<15}{'Porter':<15}{'Snowball':<15}{'Lancaster':<15}{'Lemma':<15}")
    print("-" * 75)
    for word in test_words:
        porter = processor.porter.stem(word)
        snowball = processor.snowball.stem(word)
        lancaster = processor.lancaster.stem(word)
        lemma = processor.lemmatizer.lemmatize(word, processor.get_wordnet_pos(word)) \
            if processor.lemmatizer else word
        print(f"{word:<15}{porter:<15}{snowball:<15}{lancaster:<15}{lemma:<15}")

def main():
    processor = AdvancedTextProcessor()
    demonstrate_stemmer_lemmatizer_characteristics()

    football_text = """
    Football is one of the most followed sports globally. Clubs like Real Madrid, 
    Barcelona, and Liverpool have millions of fans and rich histories. Players train 
    for hours to master passing, shooting, and dribbling. The Champions League is 
    considered the most prestigious club competition in Europe. Rivalries like El 
    Clasico between Madrid and Barca create intense excitement. Young talents like 
    Jude Bellingham and Ansu Fati are rising stars in the football world.
    """

    print("\n" + "=" * 80)
    print("FOOTBALL TEXT ANALYSIS".center(80))
    print("=" * 80)

    analysis = processor.detailed_analysis(football_text)
    stats = analysis['statistics']

    print("\nSTATISTICS:")
    for k, v in stats.items():
        if "reduction" in k:
            print(f"{k.replace('_', ' ').title()}: {v:.2f}%")
        else:
            print(f"{k.replace('_', ' ').title()}: {v}")

    print("\nCOMPARISON TABLE:")
    processor.show_comparison_table(football_text)

    diff = processor.find_differences(football_text)
    if not diff.empty:
        print("\nWORDS WITH DIFFERENT FORMS:")
        print(diff.to_string(index=False))
    else:
        print("\nNo significant differences found.")

    print("\nMOST FREQUENT FORMS:")
    for method in ['porter_stems', 'snowball_stems', 'lancaster_stems', 'lemmas']:
        freq = Counter(analysis['raw_results'][method]).most_common(5)
        print(f"{method.replace('_', ' ').title()}: {freq}")

    print("\n" + "=" * 80)
    print("RECOMMENDATION: Use lemmatization for better semantic accuracy!")
    print("=" * 80)

    print("\n\n" + "=" * 80)
    print("Developed by Harsh Rawte | Roll No: 22101A0047".center(80))
    print("=" * 80)

if __name__ == "__main__":
    main()
    