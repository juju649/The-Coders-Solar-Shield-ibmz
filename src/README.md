# Source Code

`recommend.py` — the battery recommendation function (Naomi).

Given the location of a predicted solar drop, it finds the highest-priority
ZIP codes (patients on electricity-dependent medical equipment, plus a bonus
for hospitals), picks the largest battery within about 100 km, and returns
one clear sentence for the operator. A human always approves the final decision.
