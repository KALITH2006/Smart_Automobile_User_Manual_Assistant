"""Build golden.jsonl from question specs.

Gold pages are *derived* from an evidence regex that must match the page text, so every
label is auditable and reproducible. A question fails the build if its evidence matches no
page or more than MAX_GOLD pages (ambiguous). Table-of-contents / index pages are excluded
because they list topics without answering them.

Groups: procedure (how-to), lookup (specific value / identifier), symptom (vague complaint),
unanswerable (not covered by an owner's manual; expected outcome is abstention).
"""
from __future__ import annotations

import json
import re
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).parent
MAX_GOLD = 4

# (manual, group, question, evidence regex)  -- regex is matched case-insensitively
SPECS = [
    # ---------- Suzuki Ciaz ----------
    ("suzuki_ciaz", "procedure", "How do I jump start the car using another vehicle's battery?", r"Use only a 12-volt battery to jump start"),
    ("suzuki_ciaz", "procedure", "How often should the tires be rotated?", r"Tires should be rotated every"),
    ("suzuki_ciaz", "procedure", "How do I position the jack to lift the car for a wheel change?", r"jack-head groove fits around"),
    ("suzuki_ciaz", "lookup", "What amperage is the cruise control fuse?", r"7\.5 A Cruise control"),
    ("suzuki_ciaz", "procedure", "How do I change the battery in the remote key?", r"o replace the battery of the remote"),
    ("suzuki_ciaz", "procedure", "When should the wiper blades be replaced and how?", r"If the wiper blades become brittle"),
    ("suzuki_ciaz", "symptom", "The brake warning light came on while driving, what should I do?", r"If the brake system warning light comes on while you"),
    ("suzuki_ciaz", "symptom", "A light came on with a ding sound reminding me about fuel", r"a ding sounds once to remind you to fill the fuel"),
    ("suzuki_ciaz", "lookup", "Which fluids and items should I check under the hood every week?", r"perform the following under-hood checks"),
    ("suzuki_ciaz", "procedure", "How do I check the battery electrolyte level?", r"level of the battery solution must be kept"),
    ("suzuki_ciaz", "lookup", "How should an infant be seated in the car?", r"Infant restraint - rear seat only"),
    ("suzuki_ciaz", "symptom", "My power window auto-up stopped working after the battery was disconnected", r"pinching prevention function needs to be initialized"),
    ("suzuki_ciaz", "procedure", "How do I use the windshield washer and wipers?", r"Wiper and Washer Operation"),
    ("suzuki_ciaz", "symptom", "The radio has no power and makes no sound", r"Fuse is blown\. Contact your authorized SUZUKI dealer"),
    ("suzuki_ciaz", "lookup", "Should the spare tire pressure also be checked?", r"Also check the tire pressure of the spare tire"),
    ("suzuki_ciaz", "procedure", "What can I do to get better mileage?", r"will help you improve fuel economy"),
    ("suzuki_ciaz", "procedure", "Can I tow the car with a cable if there is no tow truck?", r"towed by a towing cable"),
    ("suzuki_ciaz", "symptom", "What does the engine temperature gauge show and what is the normal range?", r"this gauge indicates the engine coolant tem"),
    ("suzuki_ciaz", "symptom", "What does the ABS warning light mean when it stays on?", r"Anti-Lock Brake System \(ABS\) Warning Light \(if equipped\)\s+65D529"),
    # ---------- BMW ----------
    ("bmw", "procedure", "How do I jump-start the car?", r"Starting aid terminals The starting aid terminal in the engine compartment acts"),
    ("bmw", "lookup", "Where can I find the correct tire inflation pressure values?", r"values can be found on the sign on the door pillar"),
    ("bmw", "symptom", "I think I have a flat tire, what should I do?", r"Actions in the event of a flat tire"),
    ("bmw", "procedure", "How do I open the fuel cap to refuel?", r"Turn the fuel cap counterclockwise"),
    ("bmw", "procedure", "How do I display the engine oil level?", r"Displaying the engine oil level"),
    ("bmw", "lookup", "What should I mix into the washer fluid reservoir?", r"Use a mixture of tap water and windshield washer concentrate"),
    ("bmw", "lookup", "Can I use water-repellent additives in the windshield washer?", r"Silicon-containing additives in the washer fluid"),
    ("bmw", "procedure", "How do I secure a child seat in the car?", r"child restraint system fits securely against the backrest"),
    ("bmw", "lookup", "How does the tire pressure monitor work?", r"The system monitors tire inflation pressure in the four mounted tires"),
    ("bmw", "symptom", "A yellow tire symbol appeared on the display, what do I do?", r"If a tire inflation pressure check is required"),
    ("bmw", "symptom", "The wipers are frozen to the windshield", r"If the wipers are frozen to the windshield"),
    ("bmw", "procedure", "How do I switch off the engine on a manual transmission car?", r"Manual transmission: switching off"),
    ("bmw", "lookup", "How big is the fuel tank?", r"Fuel tank, approx\. US gal/liters"),
    ("bmw", "procedure", "How do I fold the wipers away so I can change the blades?", r"Important, for instance when changing the wiper blades"),
    ("bmw", "lookup", "Where is the tow fitting kept?", r"The tow fitting and the onboard vehicle tool kit"),
    ("bmw", "procedure", "How do I connect a battery charger safely?", r"Only connect battery chargers for the vehicle battery"),
    ("bmw", "symptom", "The coolant looks low, do I need to check more than one place?", r"always check the coolant level of both"),
    ("bmw", "symptom", "A yellow light came on when the fuel is low", r"yellow indicator light illuminates, once the fuel reserve is reached"),
    ("bmw", "procedure", "How do I replace the battery of the vehicle key?", r"Each vehicle key contains a replaceable battery"),
]

# Not covered by an owner's manual -> gold_pages empty. `absent` terms must not appear anywhere.
UNANSWERABLE = [
    ("suzuki_ciaz", "What is the torque specification for the cylinder head bolts?", [r"cylinder head bolt"]),
    ("suzuki_ciaz", "How do I replace the timing chain?", [r"timing chain"]),
    ("suzuki_ciaz", "How do I rebuild the CVT transmission?", [r"rebuild"]),
    ("suzuki_ciaz", "How do I flash new software onto the engine control unit?", [r"flash(ing)? (new )?software", r"ECU (flash|reprogram)"]),
    ("bmw", "What is the torque specification for the cylinder head bolts?", [r"cylinder head bolt"]),
    ("bmw", "How do I replace the clutch disc?", [r"clutch disc"]),
    ("bmw", "What is the valve clearance adjustment procedure?", [r"valve clearance"]),
    ("bmw", "How do I rebuild the turbocharger?", [r"rebuild"]),
    ("bmw", "How do I recode the engine control unit with a laptop?", [r"recode the engine", r"ECU (flash|reprogram)"]),
]


# Near-domain questions an owner's manual does not answer; asked of both manuals.
SHARED_UNANSWERABLE = [
    ("How do I perform a cylinder compression test?", [r"compression test"]),
    ("How do I replace the cylinder head gasket?", [r"head gasket"]),
    ("How do I remove the engine from the vehicle?", [r"engine (removal|out of the vehicle)", r"remove the engine from"]),
    ("What are the wiring diagram colors for the ignition coil harness?", [r"wiring diagram"]),
    ("How do I reset the adaptation values using a diagnostic scan tool?", [r"scan tool"]),
    ("What is the stall speed of the torque converter?", [r"stall speed"]),
    ("How do I replace the engine oil pump?", [r"oil pump"]),
    ("What is the part number for the front brake pads?", [r"brake pad part", r"part number for the"]),
    ("How do I adjust the camshaft timing?", [r"camshaft timing", r"valve timing"]),
    ("How do I install a lift kit?", [r"lift kit"]),
    ("What is the capital of France?", [r"capital of france"]),
    ("Give me a recipe for chocolate cake", [r"chocolate"]),
]
for _m in ("suzuki_ciaz", "bmw"):
    for _q, _a in SHARED_UNANSWERABLE:
        UNANSWERABLE.append((_m, _q, _a))


def split_of(manual: str, question: str) -> str:
    """Deterministic dev/test split: tuning may only use `dev`; results are reported on `test`."""
    return "dev" if zlib.crc32(f"{manual}|{question}".encode()) % 2 == 0 else "test"


def norm(text: str) -> str:
    text = text.replace("‐\n", "").replace("‐ ", "").replace("‐", "-")
    return re.sub(r"\s+", " ", text)


def is_front_or_back_matter(manual: str, page_no: int, raw: str) -> bool:
    if len(re.findall(r"\.{6,}", raw)) >= 4:  # dotted-leader table of contents
        return True
    return manual == "bmw" and (page_no <= 8 or page_no >= 267)  # contents / index


def main() -> int:
    pages = {m: json.loads((HERE / "data" / f"{m}_pages.json").read_text(encoding="utf8"))
             for m in ("bmw", "suzuki_ciaz")}
    rows, errors = [], []
    for manual, group, question, ev in SPECS:
        hits = [i + 1 for i, raw in enumerate(pages[manual])
                if not is_front_or_back_matter(manual, i + 1, raw)
                and re.search(ev, norm(raw), re.I)]
        if not 1 <= len(hits) <= MAX_GOLD:
            errors.append(f"[{manual}] {question!r}: evidence matched pages {hits}")
        rows.append({"manual": manual, "group": group, "question": question,
                     "gold_pages": hits, "evidence": ev, "split": split_of(manual, question)})
    for manual, question, absent in UNANSWERABLE:
        found = [i + 1 for i, raw in enumerate(pages[manual])
                 if any(re.search(a, norm(raw), re.I) for a in absent)]
        if found:
            errors.append(f"[{manual}] unanswerable {question!r} appears on pages {found}")
        rows.append({"manual": manual, "group": "unanswerable", "question": question,
                     "gold_pages": [], "evidence": None, "split": split_of(manual, question)})
    if errors:
        print("\n".join(errors))
        return 1
    (HERE / "golden.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf8")
    from collections import Counter
    print(Counter((r["split"], "unanswerable" if not r["gold_pages"] else "answerable") for r in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
