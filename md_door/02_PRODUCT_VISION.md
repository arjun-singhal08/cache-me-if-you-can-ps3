# Door Product Vision

## Goal

Create a local-first application that helps a new rail engineer understand a continuous door-controller data stream, see where door cycles occur, inspect why a cycle appears abnormal, and export official-format predictions.

## Primary user

A new engineer who understands railway operations but may not know Python or machine learning.

## User questions the app should answer

- What file did I upload?
- Is the stream structurally valid?
- Where are the door cycles?
- Is this cycle opening or closing?
- Which signals changed during the cycle?
- Which cycles were classified as abnormal resistance?
- Why did the model mark this cycle as unusual?
- Can I download the prediction CSV?

## Product principles

- Visual first: show the timeline and segments before model internals.
- Explain signals in engineering terms.
- Never hide segmentation errors behind a classification probability.
- Do not invent maintenance/safety thresholds that the official data does not provide.
- The prediction is a condition-monitoring aid, not a certified maintenance decision.
- Run locally by default; cloud deployment is optional.
