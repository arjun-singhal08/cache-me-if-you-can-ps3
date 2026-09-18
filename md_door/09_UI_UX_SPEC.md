# Door UI/UX Specification

## Primary interface

Use Streamlit locally for the first working version.

## Page structure

### 1. Upload & Validation

Show:

- file name;
- row count;
- time range;
- detected sampling interval statistics;
- missing/invalid columns;
- data-quality warnings.

### 2. Stream Explorer

Interactive Plotly timeline with selectable signals:

- motor current;
- motor voltage;
- EMF;
- door position;
- open/close commands;
- opening/closing states;
- opened/locked states.

Downsample only for plotting, never for inference unless explicitly tested.

### 3. Detected Cycles

Overlay shaded cycle regions on the timeline.

Cycle table:

- cycle number;
- start/end;
- duration;
- inferred operation;
- prediction;
- confidence (optional).

### 4. Cycle Detail

Click/select one cycle to show aligned current, voltage, EMF, position, and switch states.

Provide a short explanation of what differed from typical training cycles.

### 5. Export

Download `door_predictions.csv` with official columns only.

## UX rules

- Use engineer-friendly terms before ML jargon.
- Explain abbreviations such as DCSR/DCSL/DLSR/DLSL in tooltips.
- Do not show a green/red safety verdict; labels are condition-monitoring predictions, not certified maintenance clearance.
- Clearly distinguish detected boundary from classification confidence.
