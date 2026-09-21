"""Legacy unpaired quality inference is disabled.

Use ResultsDatabase.get_results_summary with an explicit repaired assignment
manifest for coverage and cost projection. Paired inference belongs to W6.
"""
class BenchmarkAnalyzer:
    def __init__(self, *args, **kwargs):
        raise ValueError("Legacy unpaired analyzer disabled; use fixed assignment report")
