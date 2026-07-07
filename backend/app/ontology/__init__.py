from app.ontology.hierarchy import ancestors_of, descendants_of, is_a, parent_of
from app.ontology.labels import EntityLabel, RelationType

__all__ = ["EntityLabel", "RelationType", "parent_of", "ancestors_of", "descendants_of", "is_a"]
