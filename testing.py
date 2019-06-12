from flair.models import TextClassifier
from flair.data import Sentence

classifier = TextClassifier.load('best-model.pt')
sentence = Sentence("")
classifier.predict(sentence)
print(sentence.labels)
