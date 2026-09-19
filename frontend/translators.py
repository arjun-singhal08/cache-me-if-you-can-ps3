def door_assessment(abnormal, total):
    if abnormal == 0:
        return dict(level="green", title="🟢 No abnormal door resistance detected", area="Passenger door movement mechanisms", detected=f"All {total} detected door cycles were classified as Normal.", follow="Continue routine condition monitoring and scheduled inspection.")
    rate = abnormal / total * 100 if total else 0
    return dict(level="red", title="🔴 Abnormal door resistance detected", area="Passenger door movement mechanism", detected=f"{abnormal} of {total} cycles ({rate:.1f}%) showed sensor behaviour consistent with abnormal mechanical resistance.", follow="Review the flagged cycles and inspect the corresponding door mechanism for obstruction, excessive friction or other sources of resistance.")

def rail_assessment(pred):
    if pred == "Normal": return dict(level="green", title="🟢 Normal vibration signature", area="Bilateral rail / axle-box vibration", detected="The measured vibration is most consistent with the Normal class.", follow="Continue routine track condition monitoring.")
    return dict(level="red", title=f"🔴 Corrugation signature detected — {pred}", area=f"Rail {pred}", detected=f"The measured axle-box vibration contains a pattern most consistent with {pred} rail corrugation.", follow=f"Prioritize {pred} for targeted track inspection and confirm the condition using the applicable maintenance procedure.")

def acv_assessment(ranked):
    cars = [x.strip() for x in str(ranked).split("|") if x.strip()]; top = cars[0] if cars else "—"
    return dict(level="red", title=f"🔴 Car {top} ranked as primary ACV suspect", area=f"ACV system — Car {top}", detected=f"Car {top} has the strongest telemetry pattern associated with the refrigerant-leak condition relative to the other cars in this consist.", follow=f"Prioritize Car {top} for ACV inspection and refrigerant-system diagnostics.")

def shm_assessment(value):
    return dict(level="green" if value < .15 else "amber", title="Structural fatigue estimate available", area="Monitored structural stress location", detected=f"The model estimates cumulative fatigue damage D = {value:.5f}. This is a condition-monitoring estimate, not a direct percentage of remaining component life.", follow="Use the estimate together with inspection history and repeated measurements to monitor changes in accumulated fatigue.")
