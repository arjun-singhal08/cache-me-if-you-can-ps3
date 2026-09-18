# ACV Product Vision

## Product name

**ACV Leak Localisation Explorer**

## Primary user

A train or maintenance engineer who understands equipment but may not be a data scientist.

## Questions the app should answer

1. Which cars are present?
2. Which ACV parameters are available?
3. Is the telemetry usable?
4. How does each car differ from its peers?
5. Which car is most suspicious?
6. What evidence drove the ranking?

## User outcome

The engineer receives:
- workbook/data-quality summary;
- discovered car IDs;
- parameter inventory;
- time coverage and sampling summary;
- peer-comparison charts;
- full eight-car ranking;
- concise evidence for top-ranked cars;
- official CSV download.

## Principles

- Explain the data before showing a prediction.
- Compare cars relative to peers before relying on fixed absolute thresholds.
- Do not invent safety or maintenance thresholds.
- Keep the numerical core deterministic.
- Any natural-language explanation is secondary and must not alter ranking.
