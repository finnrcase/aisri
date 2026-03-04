# AI Sustainability Risk Index
(work in progress)
https://aisridata.streamlit.app/

AISRI is an index used to evaluate the sustainability risk exposure and disclosure quality of AI and AI related companies based on their publicly available environmental data.

The goal of this project is to use sustainability reports and ESG disclosures to compare AI infrastructure.

AISRI produces the following:
- A risk score (100) indicating the sustainability risk of the infrastructure
- A confidence grade (A-D) based on the quality of the disclosure publicly provided
- A structured data set of environmental related metrics for AI firms

This index aims to provide neutral benchmarking for use in comparing the environmental practices of AI firms.

### Project Motivation

AI systems rely on energy intensive data centers. As AI deployment accelerates, electricity demand and carbon emissions increase rapidly.

Yet environmental disclosures from these companies vary drastically.

AISRI combats this by:
- standardising metrics
- Evaluating transparency
- producing a comparable risk index

### Methodology
AISRI evaluates companies across five infrastructure sustainability pillars.

Energy Use:
Measures transparency around electricity consumption and infrastructure energy demand.
Example metrics:
- electricity disclosure
reporting of data center energy metrics

Infrastructure Efficiency:
Evaluates operational efficiency of compute infrastructure.
Example metrics:
- Power Usage Effectiveness (PUE)
- infrastructure efficiency reporting

Carbon Intensity:
Measures emissions associated with electricity used for infrastructure.
Example metrics:
- Scope 2 emissions intensity
- renewable energy share

Offsets and Mitigation:
Evaluates how companies address remaining emissions.
Example metrics:
- renewable procurement
- carbon offsets

Transparency:
Measures disclosure quality and reporting rigor.
Example metrics:
- third-party assurance
- reporting completeness

### Scoring System
Each company receives:

Overall Risk Score
Range: 0–100
Lower scores indicate lower sustainability risk exposure.

Confidence Score:
Reflects:
coverage of disclosed metrics
data quality
recency
external assurance

Grade
Interpretation
A -
High-quality disclosure
B - 
Good disclosure
C - 
Partial disclosure
D - 
Limited disclosure/not entered


### Data Sources

AISRI uses only publicly available disclosures, including:
- corporate sustainability reports
- environmental impact reports
- ESG disclosures
- regulatory filings

AISRI does not estimate AI emissions directly and does not infer undisclosed values.

### Future Development
Planned improvements include:
- expanding coverage to 100+ AI companies
- additional infrastructure metrics
- historical trend analysis
- API access for AISRI data
- automated sustainability report ingestion

### Disclaimer
AISRI is an independent analytical project intended for research and comparative analysis.
Scores reflect reported data and disclosed metrics and should not be interpreted as definitive environmental performance evaluations.
