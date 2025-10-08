# Insurance-Company
Automatic insurance company classifier using interpretable text similarity and rule-based matching.
Automatic Classifier for Insurance Companies
This project builds an automatic classifier for insurance companies.

The core idea is simple: it takes a list of companies (with their description, tags, sector, etc.) and assigns one or more labels from a static insurance industry taxonomy to them.

The classification is performed without supervised training — using a combination of:

Text analysis (TF-IDF) to find similarities between a company's description and the definitions of the taxonomy labels;

Simple keyword matching rules to catch obvious cases;

A confidence scoring mechanism, so each label has a measure of how well it fits.

The result is an annotated list of companies, each with its most relevant insurance labels.

The project emphasizes:

Clarity and interpretability (each classification can be explained);

Scalability (it can be run on tens of thousands of companies without issue);

Ease of adaptation (you can use any other static taxonomy, not just the insurance one).
