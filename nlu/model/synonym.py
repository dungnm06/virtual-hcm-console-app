# Mapping
SYNONYM_ID = 'id'
SYNONYM_MEANING = 'meaning'
SYNONYM_WORDS = 'words'


class SynonymSet:
    def __init__(self, set_id=0, meaning='', words=None):
        # ID
        self.id = set_id
        # Meaning
        self.meaning = meaning
        # Words in this synonym set
        if words is None:
            words = []
        self.words = words
