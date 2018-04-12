

class ConceptExtractor:
	def __init__(self, annotations, concept_net_path):
		self.annotations = annotations
		self.concept_net_path = concept_net_path
		self.concepts = []