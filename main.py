import re
import os
from pathlib import Path
import spacy
import pandas as pd
from spacy.tokens import Token
from tqdm import tqdm

def get_texts(corpus: str):
    """Return a list of the files in a corpus directory."""
    result = []
    with os.scandir(corpus) as files:
        for f in files:
            if f.is_file() and f.name.endswith('.txt'):
                result.append(f)
    
    return result

def get_metadata(filename:str, documentation: pd.DataFrame):

    # Extract FileID and strip the leading zeros
    id = filename.split('_')[0].lstrip('0')

    # Extract metadata from documentation file
    return documentation.loc[id, ['SpeakerID', 'Year', 'TrialDate', 
                                  'Gender', 'Age', 'Role', 'SocialClass1', 
                                  'SocialClass2', 'OldBaileyFile'
                                  ]]

def preprocess(text:str):

    text = text.replace('—', ' - ') # Replace em dash
    text = text.replace('–', ' - ') # Replace en dash
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) # Remove ASCII

    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    cleaned = []
    for i, current_line in enumerate(lines):

        # If at last line and there's no terminal punctuation, add full stop
        if i == len(lines) - 1:
            if not re.search(r'[.!?]$', current_line):
                current_line += '.'
            cleaned.append(current_line)
            break

        next_line = lines[i+1]
        
        # Line already ends in termanal punctuation: do nothing
        if re.search(r'[.!?]$', current_line):
            cleaned.append(current_line)
        
        # Line ends in non-terminal punctuation: do nothing
        elif re.search(r'[,:;]$', current_line):
            cleaned.append(current_line)

        # Likely continuation (ends lowercase, next starts lowercase): join with space
        elif re.search(r'[a-z]$', current_line) and re.search(r'^[a-z]', next_line):
            cleaned.append(current_line + ' ')

        # Otherwise, likely missing full stop
        else:
            cleaned.append(current_line + '. ')
    
    # Join clean lines together without seperator
    combined = " ".join(cleaned)

    combined = re.sub(r'\s+', ' ', combined)   # Remove repeated whitespace
    combined = re.sub(r'\.\.+', '.', combined) # Remove repeated fullstops
    combined = re.sub(r'\s+([.!?])', r'\1', combined) # " ." -> "."

    return combined.strip()

def find_helps(text:str, window: int=200):
    """Search a large string for an occurrence of help, and return the chunk of text containing that occurrence. This chunk will be used for deriving the variables, not for display.

    Args:
        text (str): a large string of text from the corpus (ideally preprocessed/cleaned).
        window (int): an integer specifying the chunk size, i.e. the number of characters to the left and right of HELP.

    Returns:
        A list of dictionaries containing the actual text chunk; the indices where help starts and ends in the entire file; the index of where the chunk starts in the entire file. 
    """

    # Pattern finds unhelpful, helpfully, helps, etc
    pattern = re.compile(r"\b(un)?help\w*\b", flags=re.IGNORECASE)
    examples = []

    for match in pattern.finditer(text):
        help_start, help_end = match.span() # Indices of help token

        # Define indices of context snippet
        # TODO: stop at new lines?
        start = max(0, help_start - window)
        end = min(len(text), help_end + window)

        examples.append({
            'text': text[start:end], 
            'match_span': (help_start, help_end),
            'context_start': start
        })

    return examples

def get_np_tokens(obj_token: Token):
    """Return tokens of a noun phrase, exluding clausal complements and punctuation"""

    np_tokens = [obj_token]

    for child in obj_token.children:
        if child.dep_ not in ('ccomp', 'xcomp', 'advcl', 'relcl'):

            # Don't include punctuation (e.g. 'emotionally-disturbed' should not be 3 tokens)
            non_puncts = [t for t in child.subtree if not t.is_punct]
            np_tokens.extend(non_puncts)

    return np_tokens
                    
def extract_object(token: Token, is_verb_with_comp: bool):
    """Attempt to locate the object of HELP and return a dictionary with the object's POS-tag (PRON or NP) and a list of the all the tokens that make up the object (i.e. the object's subtree). This is useful for counting how long the object is.

    Sometimes the object is stored as the nsubj of the complement. E.g. in "I helped the doctor save the patient", "the doctor" is not a dobj of HELP, but a nsubj of the complement. We check for this.

    Args:
        token: the HELP token.
        is_verb_with_comp: True if HELP is verb and has complement clause.

    Returns:
        Dictionary with empty values if no object found. Otherwise, dictionary with list of words making up the object and the object's POS-tag.
    """
    result = {'tag': 'NA', 'head': 'NA', 'len': 'NA', 'present': 'NA'}
    if not is_verb_with_comp: 
        return result
    
    dobj = None
    comp_verb = None

    # Search for direct objects and complement verbs in one loop
    for child in token.children:
        if child.dep_ == 'dobj':
            dobj = child
        elif child.dep_ in {'ccomp', 'xcomp'}:
            comp_verb = child
    
    obj = dobj
    if not obj and comp_verb: # Found a complement verb and did NOT find a direct object
            for gc in comp_verb.children:
                # Subject must occur inbetween HELP (token.i) and the complement verb (comp_verb.i)
                if gc.dep_ == 'nsubj' and token.i < gc.i < comp_verb.i:
                    obj = gc
                    break
                
    # Process object if found
    if obj is not None:
        result['tag'] = 'PRO' if obj.pos_ == 'PRON' else 'NP'
        result['head'] = obj.text
        result['len'] = len(get_np_tokens(obj))
        result['present'] = 'Yes'
        return result
    else:
        result['present'] = 'No'
        return result

def extract_subject(token: Token):
    # Default values for verbs without an overt subject
    if token.pos_ != 'VERB':
        return {'pos': 'NULL', 'head': 'NA', 'animacy': 'NA'}

    result = {'pos': 'NULL', 'head': 'NA', 'animacy': 'NA'}

    for child in token.children:
        if child.dep_ == 'nsubj':
            
            if child.pos_ == 'NOUN':
                pos = 'NP'
            elif child.lower_ == 'it':
                pos = 'IT'
            elif child.pos_ == 'PRON':
                pos = 'PRO'
            else:
                return result
            result['pos'] = pos

            result['head'] = child.lemma_
            result['animacy'] = animacy(child)

    return result

def animacy(subj: Token):

    if subj.pos_ == "PRON" and subj.lower_ != "it":
        return "Animate"

    if subj.ent_type_ in {"PERSON"}:
        return "Animate"

    # common animate nouns
    if subj.lemma_.lower() in {
        "man","woman","boy","girl","person","people",
        "gentleman","lady","child","children",
        "father","mother","brother","sister", "daughter",
        "wife", "husband", "witness", "one", "neighbour",
        "son", "friend", "land-lord", "constable",
        "sailor", "shipman", "mistress", "porter", "body",
        "maid", "butcher", "stranger", "foreman", "saxon",
        "hook", "moss", "prisoner"
    }:
        return "Animate"

    return "Inanimate"


def bare_vs_full(token: Token):
    """Classify HELP as TO, BARE, ING, INING, or NA using safer dependency rules."""

    if token.pos_ != "VERB":
        return "NA"

    max_distance = 30  # assignment rule

    for child in token.children:

        # Ignore complements that are too far away
        if child.i - token.i > max_distance:
            continue

        # INING: help in doing
        if child.dep_ == "prep" and child.lemma_ == "in":
            for gchild in child.children:
                if gchild.tag_ == "VBG":
                    if gchild.i - token.i <= max_distance:
                        return "INING"

        # Verb complements
        if child.dep_ in ("xcomp", "ccomp"):

            if child.sent != token.sent:
                continue
            if any(t.is_punct for t in token.doc[token.i:child.i]):
                continue
            if child.lemma_ in {"be", "do", "have"}:
                continue
            # ING pattern
            if child.tag_ == "VBG":
                return "ING"
            
            # Check for "to"
            has_to = any(c.lemma_ == "to" for c in child.children)

            if has_to:
                return "TO"
            else:
                return "BARE"

    return "NA"
    
def verb_lemma(token: Token):
    """Return the lemma of the complement clause."""
    if token.pos_ != 'VERB':
        return 'NA' 

    for child in token.children:
        if child.dep_ in ('ccomp', 'xcomp'):
            if child.sent != token.sent:
                continue
            if child.i - token.i > 30:
                continue
            if child.lemma in {"be", "do", "have"}:
                continue
            if any(c.dep_ == 'mark' and c.lemma_ == 'that' for c in child.children):
                continue
            return child.lemma_
    return 'NA'
        
def get_polarity(token: Token):
    """Classify polarity of HELP."""
    if token.pos_ != 'VERB':
        return 'NA'

    neg_words = {'not', "n't", 'nor', 'never', 'hardly', 'scarcely', 'barely', 'no', 'nobody', 'nothing', 'nowhere'}

    # Check for 'not only'
    has_neg = any(c.dep_ == 'neg' or c.text in neg_words for c in token.children)
    has_only = any(c.dep_ == 'advmod' and c.lemma_ == 'only' for c in token.children)

    if has_neg and has_only:
        return 'POS'
    elif has_neg and not has_only:
        return 'NEG'
    else:
        return 'POS'

def get_voice(token: Token):
    """Classify the voice of the HELP instance."""
    if token.pos_ != 'VERB':
        return "NA"

    if any(child.dep_ in ('nsubjpass', 'auxpass') for child in token.children):
        return "Passive"
    return "Active"

def horror_aequi(token: Token):
    """Check whether a non-finite marker 'to' occurs within the two words before HELP."""
    if token.pos_ != 'VERB':
        return 'NA'

    # Consider the two non-punctuation tokens immediately before HELP
    prev_tokens = []
    for i in range(token.i - 1, -1, -1):
        if not token.doc[i].is_punct:
            prev_tokens.append(token.doc[i])
            if len(prev_tokens) == 2:
                break

    if any(t.lemma_ == 'to' for t in prev_tokens):
        return 'YEStoBefore'
    return 'NOtoBefore'

def count_intervening(token: Token, is_verb_with_comp: bool):
    if not is_verb_with_comp:
        return 'NA'

    for child in token.children:

        # Cancel if we come across a 'prt', like "help up"
        if child.dep_ == 'prt':
            return 'NA'
        if child.dep_ in ("xcomp", "ccomp") and "VerbForm=Inf" in child.morph:
            if child.i - token.i > 30:
                return 'NA'

            intervening = 0
            for t in token.doc[token.i+1:child.i]:
                if t.is_punct:
                    continue
                if t.lower_ in {"to", "in"}:
                    continue
                if t.dep_ in {"advcl", "relcl"}:
                    break
                intervening += 1

            return intervening

    return 'NA'

def get_kwic(text, global_start, global_end, before_window, after_window):
    """Return concordance line for display. This operation is decoupled from the chunk taken for parsing.
    
    Args:
        text (str): The cleaned text.
        global_start (int): Index of where HELP starts in the cleaned text.
        global_end (int): Index of where HELP ends in the cleaned text.
        before_window (int): Size (in characters) of context preceding HELP.
        after_window (int): Size (in characters) of context following HELP.
    """

    start = max(0, global_start - before_window)
    end = min(len(text), global_end + after_window)
    
    left_context = text[start: global_start]
    help = text[global_start:global_end]
    right_context = text[global_end:end]

    return f"{left_context}@{help}@{right_context}"

if __name__ == "__main__":
    # Load spaCy Language object and documentation file
    nlp = spacy.load('en_core_web_lg')
    documentation = pd.read_parquet('OldBailey/Documentation.parquet')
    files = get_texts('OldBailey/Processed files/')

    # Loop over files in the corpus directory
    results = []
    for file in tqdm(files, desc="Processing files", unit='file'):
        with open(file, encoding='utf-8') as f:
            text = f.read()

        # Preprocess and find instances of HELP
        cleaned_text = preprocess(text)
        examples = find_helps(cleaned_text, window=200)

        # If we find no examples of HELP, skip the file
        if not examples:
            continue

        # Get metatata
        metadata = get_metadata(file.name, documentation)

        # Tokenise, POS-tag, dependency parse, etc
        # Using nlp.pipe basically for batch processing - much faster when we get lots of texts
        docs = nlp.pipe([e['text'] for e in examples])
        for doc, indices in zip(docs, examples):

            # Start and end indices of HELP in cleaned text
            m_start, m_end = indices['match_span']
            for token in doc:

                # Get starting index of this token in the cleaned text file
                token_global_start = token.idx + indices['context_start']

                # If token's global position is the same as the HELP instance we're targetting...
                if m_start <= token_global_start < m_end and 'help' in token.lower_:

                    subj = extract_subject(token)
                    dep_var = bare_vs_full(token)
                    is_verb_with_comp = (token.pos_ == 'VERB' and dep_var != 'NA')
                    obj = extract_object(token, is_verb_with_comp)

                    result = {
                        'KWIC': get_kwic(cleaned_text, token_global_start, m_end, 100, 100),
                        'DepVar': dep_var,
                        'HelpClass': token.pos_,
                        'HelpInflection': token.tag_,
                        'Voice': get_voice(token) if dep_var != "NA" else "NA",
                        'HorrorAequi': horror_aequi(token) if dep_var != "NA" else "NA",
                        'Polarity': get_polarity(token) if dep_var != "NA" else "NA",
                        'VerbLemma': verb_lemma(token) if dep_var != "NA" else "NA",
                        'SubjType': subj['pos'] if dep_var != "NA" else "NA",
                        'SubjHead': subj['head'] if dep_var != "NA" else "NA",
                        'SubjAnimacy': subj['animacy'] if dep_var != "NA" else "NA",
                        'ObjPresent': obj['present'] if dep_var != "NA" else "NA",
                        'ObjTag': obj['tag'] if dep_var != "NA" else "NA",
                        'ObjLength': obj['len'] if dep_var != "NA" else "NA",
                        'ObjHead': obj['head'] if dep_var != "NA" else "NA",
                        'IntervWords': count_intervening(token, is_verb_with_comp) if dep_var != "NA" else "NA",
                        'Filename': file.name,
                        **metadata.to_dict()
                    }
                    results.append(result)
                    break

    print(f"Processed {len(results)} instances of help")

    #Save CSV file
    df = pd.DataFrame(results)
    df.insert(0, 'Hit', range(1, len(df) + 1)) # Hit column
    df.to_csv('help_concordance.csv', index=False)
