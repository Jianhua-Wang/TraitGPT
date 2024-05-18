"""Prompt templates."""


PREPROCESSED_PROMPT = """Reduce the complexity of the input trait of Medical studies.
You should preprocess the input trait by removing unnecessary information, such as the content in parentheses.
If the trait is an abbreviation, please convert it to its full English name as the preprocessed trait.

For example:
input trait: Phospholipids in HDL (UKB data field 23414)
Phospholipids
input trait: CAD
Coronary Artery Disease

input trait: {input}
"""

MAP_PROMPT = """Analyze the qeury trait and the Term and Alias in the terminologies.
Output the Term and the ID by one of the following criteria:
1. The Term is most similar with the query trait
2. The query trait is belonged to the Term.

Only output Term and ID in the terminologies.
Only output one Term and ID which is the most relevant to the query trait.
"""
