"""Synthetic Dataset Generator (M5 Component 4).

Industrial documents worth benchmarking against are scarce and expensive to
label. This generator builds template-based documents with known ground
truth entities/relations, so the Benchmark Suite and Offline Evaluation
Harness always have something correct to score extraction/retrieval/
reasoning against — without needing a human annotator.

Deterministic: the same seed always produces the same document sequence
(every method draws from one `random.Random(seed)` instance), so a benchmark
run today and the same run next month operate on an identical dataset —
required for the Reproducibility guarantees this milestone asks for.
"""

import random
import uuid

from app.research.datasets.models import RecommendationOutcomeSample, SyntheticDocument, TimelineEventSample

_EQUIPMENT_PREFIX_TO_LABEL = {
    "P": "Pump",
    "M": "Motor",
    "C": "Compressor",
    "V": "Valve",
    "T": "Tank",
    "S": "Sensor",
}
_MANUFACTURERS = ["Acme Industrial", "Bharat Pumps", "Kirloskar Brothers", "Grundfos", "Siemens Energy"]
_LOCATIONS = ["Zone 1", "Zone 2", "Unit A", "Unit B", "Coke Oven Battery Area"]
_STANDARDS = ["ISO 9001", "OISD-105", "DGMS Circular 12", "OSHA 1910"]
_TECHNICIANS = ["R. Sharma", "A. Kumar", "S. Patel", "J. Singh", "M. Reddy"]
_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_FAILURE_MODES = ["bearing wear", "seal degradation", "impeller corrosion", "motor winding fault", "gasket leakage"]
_SOP_TOPICS = ["Lockout-Tagout Procedure", "Confined Space Entry", "Hot Work Permit Issuance", "Rotating Equipment Startup"]
_PERMIT_TYPES = ["Hot Work Permit", "Confined Space Permit", "Electrical Isolation Permit"]


class SyntheticDatasetGenerator:
    def __init__(self, seed: int):
        self.seed = seed
        self._rng = random.Random(seed)

    def _equipment_tag(self) -> str:
        prefix = self._rng.choice(list(_EQUIPMENT_PREFIX_TO_LABEL))
        number = self._rng.randint(100, 999)
        suffix = self._rng.choice(["", "A", "B"])
        return f"{prefix}-{number}{suffix}"

    def _label_for_tag(self, tag: str) -> str:
        return _EQUIPMENT_PREFIX_TO_LABEL.get(tag[0], "IndustrialAsset")

    def _date(self) -> str:
        return f"{self._rng.choice(_MONTHS)} {self._rng.randint(2022, 2025)}"

    def _doc_id(self) -> str:
        return str(uuid.UUID(int=self._rng.getrandbits(128)))

    def generate_maintenance_log(self, equipment_tag: str | None = None) -> SyntheticDocument:
        tag = equipment_tag or self._equipment_tag()
        technician = self._rng.choice(_TECHNICIANS)
        date = self._date()
        standard = self._rng.choice(_STANDARDS)
        text = (
            f"Maintenance Log — {tag}\n\n"
            f"On {date}, {technician} performed scheduled maintenance on {tag}. "
            f"The unit was serviced per {standard} guidelines. No anomalies were noted."
        )
        return SyntheticDocument(
            doc_id=self._doc_id(),
            doc_type="maintenance_log",
            text=text,
            ground_truth_entities=[(self._label_for_tag(tag), tag), ("Standard", standard)],
            ground_truth_relations=[(tag, "MAINTAINED_BY", f"Maintenance {date}")],
            metadata={"equipment_tag": tag, "date": date},
        )

    def generate_inspection_report(self, equipment_tag: str | None = None) -> SyntheticDocument:
        tag = equipment_tag or self._equipment_tag()
        date = self._date()
        text = (
            f"Inspection Report — {tag}\n\n"
            f"Inspection of {tag} was conducted on {date}. Visual and functional checks "
            f"were completed. The equipment was found to be in acceptable operating condition."
        )
        return SyntheticDocument(
            doc_id=self._doc_id(),
            doc_type="inspection_report",
            text=text,
            ground_truth_entities=[(self._label_for_tag(tag), tag)],
            ground_truth_relations=[(tag, "INSPECTED_BY", f"Inspection {date}")],
            metadata={"equipment_tag": tag, "date": date},
        )

    def generate_incident_report(self, equipment_tag: str | None = None) -> SyntheticDocument:
        tag = equipment_tag or self._equipment_tag()
        date = self._date()
        failure_mode = self._rng.choice(_FAILURE_MODES)
        text = (
            f"Incident Report — {tag}\n\n"
            f"On {date}, {tag} failed due to {failure_mode}. The incident was reported "
            f"and an investigation was opened to determine corrective actions."
        )
        return SyntheticDocument(
            doc_id=self._doc_id(),
            doc_type="incident_report",
            text=text,
            ground_truth_entities=[(self._label_for_tag(tag), tag), ("FailureMode", failure_mode)],
            ground_truth_relations=[(tag, "FAILED_DUE_TO", failure_mode)],
            metadata={"equipment_tag": tag, "date": date},
        )

    def generate_sop(self, topic: str | None = None) -> SyntheticDocument:
        topic_name = topic or self._rng.choice(_SOP_TOPICS)
        standard = self._rng.choice(_STANDARDS)
        text = (
            f"Standard Operating Procedure — {topic_name}\n\n"
            f"This procedure defines the steps required to safely perform {topic_name.lower()}, "
            f"in accordance with {standard}. All personnel must be trained before execution."
        )
        return SyntheticDocument(
            doc_id=self._doc_id(),
            doc_type="sop",
            text=text,
            ground_truth_entities=[("Procedure", topic_name), ("Standard", standard)],
            ground_truth_relations=[(topic_name, "USES_STANDARD", standard)],
            metadata={"topic": topic_name},
        )

    def generate_oem_manual(self, equipment_tag: str | None = None, manufacturer: str | None = None) -> SyntheticDocument:
        tag = equipment_tag or self._equipment_tag()
        manufacturer_name = manufacturer or self._rng.choice(_MANUFACTURERS)
        text = (
            f"OEM Manual — {tag}\n\n"
            f"{tag} is manufactured by {manufacturer_name}. Refer to this manual for "
            f"installation, operating limits, and recommended maintenance intervals."
        )
        return SyntheticDocument(
            doc_id=self._doc_id(),
            doc_type="oem_manual",
            text=text,
            ground_truth_entities=[(self._label_for_tag(tag), tag), ("Manufacturer", manufacturer_name)],
            ground_truth_relations=[(tag, "RELATED_TO", manufacturer_name)],
            metadata={"equipment_tag": tag, "manufacturer": manufacturer_name},
        )

    def generate_failure_report(self, equipment_tag: str | None = None) -> SyntheticDocument:
        tag = equipment_tag or self._equipment_tag()
        date = self._date()
        failure_mode = self._rng.choice(_FAILURE_MODES)
        corrective_action = f"Replace affected component and adjust {self._rng.choice(['lubrication', 'calibration', 'alignment'])} schedule"
        text = (
            f"Failure Report — {tag}\n\n"
            f"{tag} experienced a failure on {date} caused by {failure_mode}. "
            f"Recommended corrective action: {corrective_action}."
        )
        return SyntheticDocument(
            doc_id=self._doc_id(),
            doc_type="failure_report",
            text=text,
            ground_truth_entities=[(self._label_for_tag(tag), tag), ("FailureMode", failure_mode), ("CorrectiveAction", corrective_action)],
            ground_truth_relations=[(tag, "FAILED_DUE_TO", failure_mode), (failure_mode, "MITIGATED_BY", corrective_action)],
            metadata={"equipment_tag": tag, "date": date},
        )

    def generate_permit(self, permit_type: str | None = None) -> SyntheticDocument:
        permit_name = permit_type or self._rng.choice(_PERMIT_TYPES)
        location = self._rng.choice(_LOCATIONS)
        date = self._date()
        text = (
            f"{permit_name} — Issued {date}\n\n"
            f"This permit authorizes work in {location} for the duration specified below. "
            f"All safety precautions must be observed."
        )
        return SyntheticDocument(
            doc_id=self._doc_id(),
            doc_type="permit",
            text=text,
            ground_truth_entities=[("Permit", permit_name), ("Location", location)],
            ground_truth_relations=[(permit_name, "LOCATED_IN", location)],
            metadata={"permit_type": permit_name, "date": date},
        )

    def generate_equipment_history(self, equipment_tag: str | None = None, num_events: int = 5) -> list[SyntheticDocument]:
        tag = equipment_tag or self._equipment_tag()
        generators = [self.generate_maintenance_log, self.generate_inspection_report, self.generate_incident_report]
        return [self._rng.choice(generators)(equipment_tag=tag) for _ in range(num_events)]

    def generate_timeline_events(self, equipment_tag: str | None = None, num_events: int = 5) -> list[TimelineEventSample]:
        tag = equipment_tag or self._equipment_tag()
        event_types = ["maintenance", "inspection", "incident", "calibration", "shutdown", "restart"]
        events = []
        for _ in range(num_events):
            event_type = self._rng.choice(event_types)
            date = self._date()
            events.append(
                TimelineEventSample(
                    equipment_tag=tag,
                    event_type=event_type,
                    description=f"{tag} {event_type} event on {date}",
                    date=date,
                )
            )
        return events

    def generate_recommendation_outcome(self, acceptance_probability: float = 0.7) -> RecommendationOutcomeSample:
        description = f"Investigate {self._rng.choice(_FAILURE_MODES)} on {self._equipment_tag()}"
        accepted = self._rng.random() < acceptance_probability
        return RecommendationOutcomeSample(recommendation_description=description, accepted=accepted)

    def generate_dataset(self, num_documents: int, doc_types: list[str] | None = None) -> list[SyntheticDocument]:
        generator_by_type = {
            "maintenance_log": self.generate_maintenance_log,
            "inspection_report": self.generate_inspection_report,
            "incident_report": self.generate_incident_report,
            "sop": self.generate_sop,
            "oem_manual": self.generate_oem_manual,
            "failure_report": self.generate_failure_report,
            "permit": self.generate_permit,
        }
        available_types = doc_types or list(generator_by_type)
        return [generator_by_type[self._rng.choice(available_types)]() for _ in range(num_documents)]
