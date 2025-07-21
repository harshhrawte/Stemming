import re 
import unicodedata 
from typing import List, Dict, Tuple, Set 
from collections import defaultdict 
import time 

class IndicTokenizer: 
    """Custom tokenizer for Hindi and Marathi using Indic-specific rules""" 

    def __init__(self): 
        # Define Devanagari Unicode ranges 
        self.devanagari_range = r'[\u0900-\u097F]' 
        self.punctuation = r'[।॥॰\.\,\;\:\!\?\"\'\(\)\[\]\{\}\-\—\–\…\'\'\"\"]' 
        self.numbers = r'[०-९0-9]+' 
        self.english_words = r'[a-zA-Z]+' 

        # Common Hindi/Marathi conjuncts and special characters 
        self.conjuncts = ['क्ष', 'त्र', 'ज्ञ', 'श्र', 'द्व', 'द्य', 'त्त', 'न्न', 'म्म', 'ल्l'] 

        # Compound word separators 
        self.compound_separators = ['-', '–', '—', '/', '+'] 

    def tokenize(self, text: str, language: str = 'hi') -> List[str]: 
        """ 
        Tokenize text using Indic-specific rules 
        Args: 
            text: Input text 
            language: 'hi' for Hindi, 'mr' for Marathi 
        Returns: 
            List of tokens 
        """ 
        if not text: 
            return [] 

        # Normalize text 
        text = unicodedata.normalize('NFC', text) 

        # Handle compound words 
        text = self._handle_compound_words(text) 

        # Split on whitespace and punctuation 
        tokens = [] 
        current_token = "" 

        for char in text: 
            if char.isspace(): 
                if current_token: 
                    tokens.append(current_token) 
                    current_token = "" 
            elif re.match(self.punctuation, char): 
                if current_token: 
                    tokens.append(current_token) 
                    current_token = "" 
                tokens.append(char) 
            else: 
                current_token += char 

        if current_token: 
            tokens.append(current_token) 

        # Filter out empty tokens and single characters (except meaningful ones) 
        meaningful_single_chars = {'।', '॥', 'व', 'न', 'म', 'क', 'र', 'स', 'त', 'द', 'प', 'ब', 'य', 
'ल', 'ह', 'ज', 'ग', 'च', 'श', 'ष', 'थ', 'ध', 'भ', 'फ', 'ख', 'घ', 'छ', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 
'ण', 'ि ', '  ', '  ', '  ', '  ', '  ', '  ', '  ', '  ', '  ', ' ् '} 

        filtered_tokens = [] 
        for token in tokens: 
            if len(token) > 1 or token in meaningful_single_chars: 
                filtered_tokens.append(token) 

        return filtered_tokens 

    def _handle_compound_words(self, text: str) -> str: 
        """Handle compound words by preserving meaningful separators""" 
        for sep in self.compound_separators: 
            # Don't split if it's likely a compound word 
            text = re.sub(f'([{self.devanagari_range}]){re.escape(sep)}([{self.devanagari_range}])', 
                         r'\1 \2', text) 
        return text 

class ManualStemmer: 
    """Manual rule-based stemmer for Hindi and Marathi""" 

    def __init__(self): 
        self.hindi_rules = self._create_hindi_rules() 
        self.marathi_rules = self._create_marathi_rules() 

    def _create_hindi_rules(self) -> List[Tuple[str, str]]: 
        """Create 50 manual stemming rules for Hindi""" 
        rules = [ 
            # Plural suffixes 
            ('    $', ''),  # लड़क   -> लड़क 
            ('ओ $', ''),  # बच्   -> बच् 
            ('    $', ''),   # लड़कें -> लड़क 
            ('    $', ''),   # औरतें -> औरत 
            ('ि य  $', '  '),  # लड़िकय   -> लड़क  
            ('  ओ $', '  '),   # आदिमय   -> आदम  
            ('  ओ $', '  '),   # बहुओ  -> बहू 

            # Verb forms 
            ('त $', ''),   # ख त  -> ख  
            ('त $', ''),   # ख त  -> ख  
            ('त $', ''),   # ख त  -> ख  
            ('न $', ''),   # ख न  -> ख  
            ('न $', ''),   # ख न  -> ख  
            ('न $', ''),   # ख न  -> ख  
            ('य $', ''),   # िकय  -> िक 
            ('य $', ''),   # िकय  -> िक 
            ('य $', ''),   # गय  -> ग 
            ('  ग $', ''),  # ज एग  -> ज  
            ('  ग $', ''),  # ज एग  -> ज  
            ('    ग $', ''),  # ज ए ग  -> ज  
            ('    ग $', ''),  # कर ग  -> कर 
            ('    ग $', ''),  # कर ग  -> कर 
            ('    ग $', ''),  # कर ग  -> कर 

            # Adjective forms 
            ('व न$', ''),  # ग णव न -> ग ण 
            ('म न$', ''),  # ब द्धिम न -> ब द्धि 
            ('द र$', ''),  # िजम्म द र -> िजम्म  
            ('क र$', ''),  # स स्क र -> स स् 
            ('ह र$', ''),  # पहनह र -> पहन 
            ('व ल $', ''),  # पढ़न व ल  -> पढ़न  
            ('व ल $', ''),  # पढ़न व ल  -> पढ़न  
            ('व ल $', ''),  # पढ़न व ल  -> पढ़न  

            # Case markers 
            ('स $', ''),   # घर स  -> घर 
            ('में$', ''),   # घर में -> घर 
            ('पर$', ''),   # म ज पर -> म ज 
            ('क $', ''),   # र म क  -> र म 
            ('क $', ''),   # र म क  -> र म 
            ('क $', ''),   # र म क  -> र म 
            ('क $', ''),   # र म क  -> र म 
            ('प $', ''),   # घर प  -> घर 
            ('तक$', ''),  # यह   तक -> यह   

            # Diminutive and augmentative 
            ('ज $', ''),   # प प ज  -> प प  
            ('स हब$', ''),  # र मस हब -> र म 
            ('ब ब $', ''),  # र मब ब  -> र म 
            ('ज न$', ''),  # ब ट ज न -> ब ट  

            # Abstract noun suffixes 
            ('त $', ''),   # मध रत  -> मध र 
            ('त्व$', ''),   # मन ष्यत्व -> मन ष्य 
            ('पन$', ''),   # बचपन -> बच 
            ('आहट$', ''),  # घबर हट -> घबर  
            ('आवट$', ''),  # सज वट -> सज  
            ('इयत$', ''),  # इ स िनयत -> इ स न 

            # Tense markers 
            ('थ $', ''),   # ज त  थ  -> ज त  
            ('थ $', ''),   # ज त  थ  -> ज त  
            ('थ $', ''),   # ज त  थ  -> ज त  
            ('हू $', ''),   # ज त  हू  -> ज त  
            ('हैं$', ''),   # ज त  हैं -> ज त  
            ('ह $', ''),   # ज त  ह  -> ज त  
        ] 
        return rules 

    def _create_marathi_rules(self) -> List[Tuple[str, str]]: 
        """Create 50 manual stemming rules for Marathi""" 
        rules = [ 
            # Plural suffixes 
            ('    न $', ''),  # म ल  न  -> म ल 
            ('    न $', ''),  # म ल  न  -> म ल 
            ('    च $', ''),  # म ल  च  -> म ल 
            ('    च $', ''),  # म ल  च  -> म ल 
            ('    च्य $', ''),  # म ल  च्य  -> म ल 
            ('    मध्य $', ''),  # म ल  मध्य  -> म ल 
            ('    स ठ $', ''),  # म ल  स ठ  -> म ल 

            # Verb forms 
            ('त $', ''),   # करत  -> कर 
            ('त $', ''),   # करत  -> कर 
            ('त $', ''),   # करत  -> कर 
            ('त त$', ''),  # करत त -> कर 
            ('ल $', ''),   # ग ल  -> ग 
            ('ल $', ''),   # ग ल  -> ग 
            ('ल $', ''),   # ग ल  -> ग 
            ('ल $', ''),   # ग ल  -> ग 
            ('  ल$', ''),   # ज ईल -> ज  
            ('  ल$', ''),   # कर ल -> कर 
            ('  य $', ''),  # करय  -> कर 
            ('  यच $', ''),  # कर यच  -> कर 
            ('  यच $', ''),  # कर यच  -> कर 
            ('  यच $', ''),  # कर यच  -> कर 

            # Case markers 
            ('ल $', ''),   # घरल  -> घर 
            ('च $', ''),   # घरच  -> घर 
            ('च $', ''),   # घरच  -> घर 
            ('च्य $', ''),  # घरच्य  -> घर 
            ('मध्य $', ''),  # घरमध्य  -> घर 
            ('वर$', ''),   # ट बलवर -> ट बल 
            ('ख ल $', ''),  # ट बलख ल  -> ट बल 
            ('जवळ$', ''),  # घरजवळ -> घर 
            ('प ढ $', ''),  # घरप ढ  -> घर 
            ('म ग $', ''),  # घरम ग  -> घर 
            ('श $', ''),   # र मश  -> र म 
            ('कड $', ''),  # र मकड  -> र म 
            ('स ठ $', ''),  # र मस ठ  -> र म 

            # Adjective forms 
            ('व न$', ''),  # ग णव न -> ग ण 
            ('म न$', ''),  # ब द्धिम न -> ब द्धि 
            ('द र$', ''),  # जब बद र -> जब ब 
            ('क र$', ''),  # स स्क र -> स स् 
            ('ह र$', ''),  # पहनह र -> पहन 
            ('व ल $', ''),  # घ ण र  -> घ  
            ('ण र $', ''),  # करण र  -> कर 
            ('ण र $', ''),  # करण र  -> कर 
            ('ण र $', ''),  # करण र  -> कर 

            # Abstract noun suffixes 
            ('त $', ''),   # मध रत  -> मध र 
            ('त्व$', ''),   # मन ष्यत्व -> मन ष्य 
            ('पण $', ''),  # लह नपण  -> लह न 
            ('आई$', ''),  # म ठ ई -> म ठ 
            ('इक$', ''),  # स प्त िहक -> सप्त ह 

            # Honorific suffixes 
            ('ज $', ''),   # ब ब ज  -> ब ब  
            ('स ह ब$', ''),  # र मस ह ब -> र म 
            ('र व$', ''),  # र मर व -> र म 
            ('आज $', ''),  # आज  -> आज 
            ('क क $', ''),  # र मक क  -> र म 

            # Tense markers 
            ('ह त $', ''),  # ज त ह त  -> ज त 
            ('ह त $', ''),  # ज त ह त  -> ज त 
            ('ह त $', ''),  # ज त ह त  -> ज त 
            ('आह $', ''),  # ज त  आह  -> ज त  
            ('आह त$', ''),  # ज त त आह त -> ज त त 
        ] 
        return rules 

    def stem(self, word: str, language: str) -> str: 
        """ 
        Apply manual stemming rules 
        Args: 
            word: Input word 
            language: 'hi' for Hindi, 'mr' for Marathi 
        Returns: 
            Stemmed word 
        """ 
        if not word: 
            return word 

        rules = self.hindi_rules if language == 'hi' else self.marathi_rules 

        # Apply rules in order 
        for pattern, replacement in rules: 
            if re.search(pattern, word): 
                stemmed = re.sub(pattern, replacement, word) 
                if stemmed and stemmed != word: 
                    return stemmed 

        return word 

class PretrainedStemmer: 
    """Wrapper for pretrained stemming models""" 

    def __init__(self): 
        # Simulated pretrained models (in practice, these would be loaded from files) 
        self.hindi_model = self._load_hindi_model() 
        self.marathi_model = self._load_marathi_model() 

    def _load_hindi_model(self) -> Dict[str, str]: 
        """Simulate loading a pretrained Hindi stemmer""" 
        # This would typically load from a file or model 
        return { 
            'लड़क  ': 'लड़क ', 
            'लड़िकय  ': 'लड़क ', 
            'ख त ': 'ख ', 
            'ख त ': 'ख ', 
            'ख त ': 'ख ', 
            'करत ': 'कर', 
            'करत ': 'कर', 
            'करत ': 'कर', 
            'ज त ': 'ज ', 
            'ज त ': 'ज ', 
            'ज त ': 'ज ', 
            'आदम ': 'आदम ', 
            'आदिमय  ': 'आदम ', 
            'औरत': 'औरत', 
            'औरतें': 'औरत', 
            'बच् ': 'बच् ', 
            'बच्  ': 'बच् ', 
            'ग णव न': 'ग ण', 
            'ब द्धिम न': 'ब द्धि', 
            'िजम्म द र': 'िजम्म द र ', 
            'स स्क र': 'स स्क र', 
            'मध रत ': 'मध र', 
            'मन ष्यत्व': 'मन ष्य', 
            'बचपन': 'बच् ', 
            'इ स िनयत': 'इ स न', 
            'घबर हट': 'घबर ', 
            'सज वट': 'सज ', 
            'प प ज ': 'प प ', 
            'र मस हब': 'र म', 
            'र मब ब ': 'र म', 
            'गय ': 'ज ', 
            'िकय ': 'कर', 
            'िकय ': 'कर', 
            'ज एग ': 'ज ', 
            'ज एग ': 'ज ', 
            'ज ए ग ': 'ज ', 
            'कर ग ': 'कर', 
            'कर ग ': 'कर', 
            'कर ग ': 'कर', 
            'थ ': 'ह ', 
            'थ ': 'ह ', 
            'थ ': 'ह ', 
            'हू ': 'ह ', 
            'हैं': 'ह ', 
            'पढ़न व ल ': 'पढ़', 
            'पढ़न व ल ': 'पढ़', 
            'पढ़न व ल ': 'पढ़' 
        } 

    def _load_marathi_model(self) -> Dict[str, str]: 
        """Simulate loading a pretrained Marathi stemmer""" 
        return { 
            'म ल  न ': 'म ल', 
            'म ल  न ': 'म ल', 
            'म ल  च ': 'म ल', 
            'म ल  च ': 'म ल', 
            'म ल  च्य ': 'म ल', 
            'करत ': 'कर', 
            'करत ': 'कर', 
            'करत ': 'कर', 
            'करत त': 'कर', 
            'ग ल ': 'ज ', 
            'ग ल ': 'ज ', 
            'ग ल ': 'ज ', 
            'ग ल ': 'ज ', 
            'ज ईल': 'ज ', 
            'कर ल': 'कर', 
            'करय ': 'कर', 
            'कर यच ': 'कर', 
            'कर यच ': 'कर', 
            'कर यच ': 'कर', 
            'घरल ': 'घर', 
            'घरच ': 'घर', 
            'घरच ': 'घर', 
            'घरच्य ': 'घर', 
            'घरमध्य ': 'घर', 
            'ट बलवर': 'ट बल', 
            'ट बलख ल ': 'ट बल', 
            'घरजवळ': 'घर', 
            'घरप ढ ': 'घर', 
            'घरम ग ': 'घर', 
            'र मश ': 'र म', 
            'र मकड ': 'र म', 
            'र मस ठ ': 'र म', 
            'ग णव न': 'ग ण', 
            'ब द्धिम न': 'ब द्धि', 
            'जब बद र': 'जब ब', 
            'स स्क र': 'स स्क र', 
            'करण र ': 'कर', 
            'करण र ': 'कर', 
            'करण र ': 'कर', 
            'मध रत ': 'मध र', 
            'मन ष्यत्व': 'मन ष्य', 
            'लह नपण ': 'लह न', 
            'म ठ ई': 'म ठ', 
            'स प्त िहक': 'सप्त ह', 
            'ब ब ज ': 'ब ब ', 
            'र मस ह ब': 'र म', 
            'र मर व': 'र म', 
            'आज ': 'आज', 
            'र मक क ': 'र म', 
            'ह त ': 'असण ', 
            'ह त ': 'असण ', 
            'ह त ': 'असण ', 
            'आह ': 'असण ', 
            'आह त': 'असण ' 
        } 

    def stem(self, word: str, language: str) -> str: 
        """ 
        Apply pretrained stemming model 
        Args: 
            word: Input word 
            language: 'hi' for Hindi, 'mr' for Marathi 
        Returns: 
            Stemmed word 
        """ 
        if not word: 
            return word 

        model = self.hindi_model if language == 'hi' else self.marathi_model 
        return model.get(word, word) 

class MultilingualProcessor: 
    """Main processor combining tokenization and stemming""" 

    def __init__(self): 
        self.tokenizer = IndicTokenizer() 
        self.manual_stemmer = ManualStemmer() 
        self.pretrained_stemmer = PretrainedStemmer()

    def process_text(self, text: str, language: str) -> Dict: 
        """ 
        Process text through tokenization and both stemming approaches 
        Args: 
            text: Input text 
            language: 'hi' for Hindi, 'mr' for Marathi 
        Returns: 
            Dictionary with results 
        """ 
        # Tokenization 
        start_time = time.time() 
        tokens = self.tokenizer.tokenize(text, language) 
        tokenization_time = time.time() - start_time 

        # Manual stemming 
        start_time = time.time() 
        manual_stems = [self.manual_stemmer.stem(token, language) for token in tokens] 
        manual_stemming_time = time.time() - start_time 

        # Pretrained stemming 
        start_time = time.time() 
        pretrained_stems = [self.pretrained_stemmer.stem(token, language) for token in tokens] 
        pretrained_stemming_time = time.time() - start_time 

        return { 
            'original_text': text, 
            'language': language, 
            'tokens': tokens, 
            'token_count': len(tokens), 
            'manual_stems': manual_stems, 
            'pretrained_stems': pretrained_stems,
            'tokenization_time': tokenization_time, 
            'manual_stemming_time': manual_stemming_time, 
            'pretrained_stemming_time': pretrained_stemming_time
        } 

    def compare_stemmers(self, test_words: List[str], language: str) -> Dict: 
        """ 
        Compare manual and pretrained stemming approaches 
        Args: 
            test_words: List of words to test 
            language: 'hi' for Hindi, 'mr' for Marathi 
        Returns: 
            Comparison results 
        """ 
        results = { 
            'word': [], 
            'manual_stem': [], 
            'pretrained_stem': [], 
            'agreement': [], 
            'manual_reduction': [], 
            'pretrained_reduction': [] 
        } 

        for word in test_words: 
            manual_stem = self.manual_stemmer.stem(word, language) 
            pretrained_stem = self.pretrained_stemmer.stem(word, language) 

            results['word'].append(word) 
            results['manual_stem'].append(manual_stem) 
            results['pretrained_stem'].append(pretrained_stem) 
            results['agreement'].append(manual_stem == pretrained_stem) 
            results['manual_reduction'].append(len(word) - len(manual_stem)) 
            results['pretrained_reduction'].append(len(word) - len(pretrained_stem)) 

        # Calculate statistics 
        agreement_rate = sum(results['agreement']) / len(results['agreement']) 
        avg_manual_reduction = sum(results['manual_reduction']) / len(results['manual_reduction']) 
        avg_pretrained_reduction = sum(results['pretrained_reduction']) / len(results['pretrained_reduction']) 

        return { 
            'detailed_results': results, 
            'agreement_rate': agreement_rate, 
            'avg_manual_reduction': avg_manual_reduction, 
            'avg_pretrained_reduction': avg_pretrained_reduction 
        } 

# Example usage and testing 
def main(): 
    processor = MultilingualProcessor() 

    # Test Hindi text - VIT Mumbai description 
    hindi_text = """व भ क ा ट क न ल ज म बई (VIT म बई) भ रत क  एक प रख य त तक न क स स थ न ह । 
यह मह र ष ट र क म बई शहर म स थ त ह और इ ज न यर ग, व ज ञ न, प रब धन और ड ज इन क  क ष त र म 
उ चतम श क ष प रद न करत  ह । स स थ न क  प रय स ह छ त र   क  समग र व क स क  ल ए 
अन कदम उठ न  ह । यह  नव श क षण और अन स ध न क  म ध यम स सम ज क  सम य ओ  क  
सम ध न ख जन  म अग रण रह  ह ।"""

    # Test Marathi text - VIT Mumbai description 
    marathi_text = """व भ क ा ट क न ल ज म बई (VIT म बई) ह  भ रत त ल एक प रख य त तक न क 
संस थ न आह . ह  मह र ष ट र य त ल म बई शहर त वसत  आ ण त तक न क श क षण, व ज ञ न, 
व यवस थ पन आ ण ड ज इन य  क ष त त ल उत तम श क षण प उपलब ध कर त . ह  संस थ न 
व द य र्थ य न च  स व गत व क स स ठ  अनेक प वल  उचलत . नव नव त त ज ञ न आ ण संश धन 
म ध यम त न सम ज य च य  सम य  स म ण य  उप य शोध य स ठ  अग रसर आह ."""

    print("=== HINDI TEXT PROCESSING (VIT Mumbai) ===") 
    hindi_results = processor.process_text(hindi_text, 'hi') 
    print(f"Original text length: {len(hindi_results['original_text'])} characters") 
    print(f"Token count: {hindi_results['token_count']}") 
    print(f"First 10 tokens: {hindi_results['tokens'][:10]}") 
    print(f"First 10 manual stems: {hindi_results['manual_stems'][:10]}") 
    print(f"First 10 pretrained stems: {hindi_results['pretrained_stems'][:10]}")
    print(f"Tokenization time: {hindi_results['tokenization_time']:.4f}s") 
    print(f"Manual stemming time: {hindi_results['manual_stemming_time']:.4f}s") 
    print(f"Pretrained stemming time: {hindi_results['pretrained_stemming_time']:.4f}s")

    print("\n=== MARATHI TEXT PROCESSING (VIT Mumbai) ===") 
    marathi_results = processor.process_text(marathi_text, 'mr') 
    print(f"Original text length: {len(marathi_results['original_text'])} characters") 
    print(f"Token count: {marathi_results['token_count']}") 
    print(f"First 10 tokens: {marathi_results['tokens'][:10]}") 
    print(f"First 10 manual stems: {marathi_results['manual_stems'][:10]}") 
    print(f"First 10 pretrained stems: {marathi_results['pretrained_stems'][:10]}")
    print(f"Tokenization time: {marathi_results['tokenization_time']:.4f}s") 
    print(f"Manual stemming time: {marathi_results['manual_stemming_time']:.4f}s") 
    print(f"Pretrained stemming time: {marathi_results['pretrained_stemming_time']:.4f}s")

    # Analysis of unique tokens vs stems 
    print("\n=== HINDI VOCABULARY ANALYSIS ===") 
    hindi_unique_tokens = set(hindi_results['tokens']) 
    hindi_unique_manual_stems = set(hindi_results['manual_stems']) 
    hindi_unique_pretrained_stems = set(hindi_results['pretrained_stems']) 

    print(f"Unique tokens: {len(hindi_unique_tokens)}") 
    print(f"Unique manual stems: {len(hindi_unique_manual_stems)}")
    print(f"Unique pretrained stems: {len(hindi_unique_pretrained_stems)}")
    print(f"Vocabulary reduction (manual): {len(hindi_unique_tokens) - len(hindi_unique_manual_stems)}")
    print(f"Vocabulary reduction (pretrained): {len(hindi_unique_tokens) - len(hindi_unique_pretrained_stems)}")

    print("\n=== MARATHI VOCABULARY ANALYSIS ===")
    marathi_unique_tokens = set(marathi_results['tokens'])
    marathi_unique_manual_stems = set(marathi_results['manual_stems'])
    marathi_unique_pretrained_stems = set(marathi_results['pretrained_stems'])

    print(f"Unique tokens: {len(marathi_unique_tokens)}")
    print(f"Unique manual stems: {len(marathi_unique_manual_stems)}")
    print(f"Unique pretrained stems: {len(marathi_unique_pretrained_stems)}")
    print(f"Vocabulary reduction (manual): {len(marathi_unique_tokens) - len(marathi_unique_manual_stems)}")
    print(f"Vocabulary reduction (pretrained): {len(marathi_unique_tokens) - len(marathi_unique_pretrained_stems)}")

    # Sample comparison of stems
    print("\n=== SAMPLE STEM COMPARISON (Hindi) ===")
    print(f"{'Token':<20} {'Manual':<20} {'Pretrained':<20} {'Reduction':<10}")
    print("-" * 75)
    for i, (token, manual, pretrained) in enumerate(zip(
        hindi_results['tokens'][:15], 
        hindi_results['manual_stems'][:15], 
        hindi_results['pretrained_stems'][:15]
    )):
        reduction = len(token) - len(manual)
        print(f"{token:<20} {manual:<20} {pretrained:<20} {reduction:<10}")

    print("\n=== SAMPLE STEM COMPARISON (Marathi) ===")
    print(f"{'Token':<20} {'Manual':<20} {'Pretrained':<20} {'Reduction':<10}")
    print("-" * 75)
    for i, (token, manual, pretrained) in enumerate(zip(
        marathi_results['tokens'][:15], 
        marathi_results['manual_stems'][:15], 
        marathi_results['pretrained_stems'][:15]
    )):
        reduction = len(token) - len(manual)
        print(f"{token:<20} {manual:<20} {pretrained:<20} {reduction:<10}")

    # Enhanced word comparison with domain-specific terms
    print("\n=== ENHANCED HINDI STEMMER COMPARISON ===")
    hindi_test_words = [
        'इंडियन', 'प्रीमियर', 'लीग', 'क्रिकेट', 'टीम', 'खिलाड़ी', 'टूर्नामेंट', 'विकेट', 'गेंदबाज़',
        'बल्लेबाज़', 'अंपायर', 'टी-20', 'रन', 'ओवर', 'फ़ील्डिंग', 'कप्तान', 'चैंपियन', 'मुंबई',
        'चेन्नई',
        'खिताब', 'प्रतिस्पर्धा', 'आईपीएल', 'प्लेऑफ़', 'ऑक्शन', 'स्टेडियम'
    ]

    hindi_comparison = processor.compare_stemmers(hindi_test_words, 'hi')
    print(f"Agreement rate: {hindi_comparison['agreement_rate']:.2%}")
    print(f"Average manual reduction: {hindi_comparison['avg_manual_reduction']:.1f} characters")
    print(f"Average pretrained reduction: {hindi_comparison['avg_pretrained_reduction']:.1f} characters")

    print("\n=== ENHANCED MARATHI STEMMER COMPARISON ===")
    marathi_test_words = [
        'आयपीएल', 'क्रिकेट', 'सामना', 'संघ', 'खेळाडू', 'गोलंदाज', 'फलंदाज', 'कर्णधार',
        'विरुद्ध', 'बाद',
        'सामना', 'गड', 'फील्डिंग', 'धाव', 'चेंडू', 'विकेट', 'खेळपट्टी', 'साखळी', 'उपाध्यक्ष', 'अंतिम',
        'विजेता', 'मुंबई', 'चेन्नई', 'स्पर्धा', 'प्रशिक्षक', 'संघटना'
    ]
    
    marathi_comparison = processor.compare_stemmers(marathi_test_words, 'mr')
    print(f"Agreement rate: {marathi_comparison['agreement_rate']:.2%}")
    print(f"Average manual reduction: {marathi_comparison['avg_manual_reduction']:.1f} characters")
    print(f"Average pretrained reduction: {marathi_comparison['avg_pretrained_reduction']:.1f} characters")

    # Detailed comparison table with enhanced test words
    print("\n=== DETAILED HINDI COMPARISON (Domain-specific Terms) ===")
    print(f"{'Word':<20} {'Manual':<20} {'Pretrained':<20} {'Agreement':<12} {'Reduction':<10}")
    print("-" * 85)
    for i, word in enumerate(hindi_test_words):
        manual = hindi_comparison['detailed_results']['manual_stem'][i]
        pretrained = hindi_comparison['detailed_results']['pretrained_stem'][i]
        agreement = "✓" if hindi_comparison['detailed_results']['agreement'][i] else "✗"
        reduction = hindi_comparison['detailed_results']['manual_reduction'][i]
        print(f"{word:<20} {manual:<20} {pretrained:<20} {agreement:<12} {reduction:<10}")

    print("\n=== DETAILED MARATHI COMPARISON (Domain-specific Terms) ===")
    print(f"{'Word':<20} {'Manual':<20} {'Pretrained':<20} {'Agreement':<12} {'Reduction':<10}")
    print("-" * 85)
    for i, word in enumerate(marathi_test_words):
        manual = marathi_comparison['detailed_results']['manual_stem'][i]
        pretrained = marathi_comparison['detailed_results']['pretrained_stem'][i]
        agreement = "✓" if marathi_comparison['detailed_results']['agreement'][i] else "✗"
        reduction = marathi_comparison['detailed_results']['manual_reduction'][i]
        print(f"{word:<20} {manual:<20} {pretrained:<20} {agreement:<12} {reduction:<10}")

    # Performance and effectiveness analysis
    print("\n=== PERFORMANCE AND EFFECTIVENESS ANALYSIS ===")
    print("Manual Rule-Based Stemmer:")
    print("  Pros: Fast execution, predictable results, language-specific rules")
    print("  Cons: Limited coverage, may over-stem or under-stem")
    print(f"  Hindi Performance: {hindi_comparison['agreement_rate']:.1%} agreement, {hindi_comparison['avg_manual_reduction']:.1f} avg reduction")
    print(f"  Marathi Performance: {marathi_comparison['agreement_rate']:.1%} agreement, {marathi_comparison['avg_manual_reduction']:.1f} avg reduction")

    print("\nPretrained Model Stemmer:")
    print("  Pros: Better context understanding, handles exceptions well")
    print("  Cons: Slower execution, requires training data, may not cover new terms")
    print(f"  Hindi Performance: {hindi_comparison['avg_pretrained_reduction']:.1f} avg reduction")
    print(f"  Marathi Performance: {marathi_comparison['avg_pretrained_reduction']:.1f} avg reduction")

    # Recommendation
    print("\n=== RECOMMENDATION ===")
    if hindi_comparison['agreement_rate'] > 0.7:
        print("HIGH AGREEMENT: Both approaches show similar results, manual rules are sufficient")
    elif hindi_comparison['agreement_rate'] > 0.5:
        print("MODERATE AGREEMENT: Consider hybrid approach combining both methods")
    else:
        print("LOW AGREEMENT: Pretrained model likely more accurate for complex cases")

    print(f"For news aggregation platform: Use manual rules for speed, pretrained for accuracy")
    print(f"Recommended approach: Hybrid system with manual rules as baseline + pretrained for complex terms")

if __name__ == "__main__":
    main()