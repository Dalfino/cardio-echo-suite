"""FHIR R4 → DICOM Structured Report (DICOM-SR) converter.

Many hospital PACS systems speak DICOM-SR, not FHIR. This module converts
a FHIR R4 DiagnosticReport or Observation Bundle into a DICOM-SR document
that can be stored in PACS alongside the source imaging study.

Output format: pydicom Dataset for SOPClassUID = Comprehensive SR (1.2.840.10008.5.1.4.1.1.88.33)

Usage:
    from cardio_echo_core.dicom_sr import fhir_to_dicom_sr

    sr_ds = fhir_to_dicom_sr(
        fhir_report=fhir_diagnostic_report_dict,
        patient_id="PATIENT-001",
        patient_name="TEST^PATIENT",
        study_instance_uid="1.2.3.4.5",
        series_instance_uid="1.2.3.4.6",
        sop_instance_uid="1.2.3.4.7",
    )
    sr_ds.save_as("/tmp/report_sr.dcm")
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid, UID

# Comprehensive SR Document Storage SOP Class UID
# (1.2.840.10008.5.1.4.1.1.88.33)
COMPREHENSIVE_SR_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.88.33"


# ----------------------------------------------------------------------
# Code sequences for common findings
# ----------------------------------------------------------------------

# LOINC codes used in echo reports
LOINC_CODES = {
    "10230-1": "Ejection fraction",
    "26520-0": "Echocardiography study",
    "11524-6": "EKG study",
    "44967-8": "PR interval",
    "44577-8": "QRS duration",
    "44996-7": "QT interval",
    "44978-5": "QTc interval",
    "44980-1": "RR interval",
    "8867-4":  "Heart rate",
}

# SNOMED CT codes for rhythms / findings
SNOMED_CODES = {
    "426783004": "Sinus rhythm",
    "164873001": "Atrial fibrillation",
    "164864005": "Atrial flutter",
    "71908006":  "Ventricular tachycardia",
    "426788005": "Ventricular fibrillation",
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _make_code_sequence(code: str, system: str, meaning: str) -> Dataset:
    """Build a DICOM CodeSequenceItem."""
    if "loinc.org" in system:
        scheme = "DCM"
        ctx = "1.2.840.10008.2.16.4"  # LOINC coding scheme
    elif "snomed.info" in system:
        scheme = "SCT"
        ctx = "1.2.840.10008.2.16.8"  # SNOMED CT
    else:
        scheme = "DCM"
        ctx = ""

    item = Dataset()
    item.CodeValue = code
    item.CodingSchemeDesignator = scheme
    if ctx:
        item.CodingSchemeUID = ctx
    item.CodeMeaning = meaning
    return item


def _make_text_content(value: str, relationship: str = "CONTAINS") -> Dataset:
    """Build a TEXT content item."""
    item = Dataset()
    item.RelationshipType = relationship
    item.ValueType = "TEXT"
    item.ConceptNameCodeSequence = [_make_code_sequence("113012", "DCM", "Observation Description")]
    item.TextValue = value[:1024] if value else ""
    return item


def _make_num_content(
    value: float, unit: str, code: str, system: str, meaning: str,
    relationship: str = "CONTAINS",
) -> Dataset:
    """Build a NUM content item."""
    item = Dataset()
    item.RelationshipType = relationship
    item.ValueType = "NUM"
    item.ConceptNameCodeSequence = [_make_code_sequence(code, system, meaning)]
    measured = Dataset()
    measured.NumericValue = float(value)
    if unit == "%":
        measured.MeasurementUnitsCodeSequence = [_make_code_sequence(
            "percent", "UCUM", "percent"
        )]
    elif unit == "ms":
        measured.MeasurementUnitsCodeSequence = [_make_code_sequence(
            "ms", "UCUM", "milliseconds"
        )]
    elif unit in ("mL", "ml"):
        measured.MeasurementUnitsCodeSequence = [_make_code_sequence(
            "mL", "UCUM", "milliliters"
        )]
    elif unit == "cm":
        measured.MeasurementUnitsCodeSequence = [_make_code_sequence(
            "cm", "UCUM", "centimeters"
        )]
    elif unit in ("/min", "bpm"):
        measured.MeasurementUnitsCodeSequence = [_make_code_sequence(
            "/min", "UCUM", "beats per minute"
        )]
    else:
        measured.MeasurementUnitsCodeSequence = [_make_code_sequence(
            "1", "UCUM", "no units"
        )]
    item.MeasuredValueSequence = [measured]
    return item


def _make_code_content(
    code: str, system: str, meaning: str,
    name_code: str = "113041",
    name_system: str = "DCM",
    name_meaning: str = "Finding",
    relationship: str = "CONTAINS",
) -> Dataset:
    """Build a CODE content item."""
    item = Dataset()
    item.RelationshipType = relationship
    item.ValueType = "CODE"
    item.ConceptNameCodeSequence = [_make_code_sequence(name_code, name_system, name_meaning)]
    item.ConceptCodeSequence = [_make_code_sequence(code, system, meaning)]
    return item


# ----------------------------------------------------------------------
# Main converter
# ----------------------------------------------------------------------

def fhir_to_dicom_sr(
    fhir_report: Dict[str, Any],
    patient_id: str = "UNKNOWN",
    patient_name: str = "UNKNOWN^PATIENT",
    patient_birth_date: Optional[str] = None,
    patient_sex: Optional[str] = None,
    study_instance_uid: Optional[str] = None,
    series_instance_uid: Optional[str] = None,
    sop_instance_uid: Optional[str] = None,
    accession_number: Optional[str] = None,
    study_description: str = "AI pre-read",
    study_date: Optional[str] = None,
    manufacturer: str = "cardio-echo-suite",
    manufacturer_model_name: str = "cardio-echo-suite v0.1",
    software_versions: str = "0.1.0",
) -> FileDataset:
    """Convert a FHIR R4 DiagnosticReport or Bundle to a DICOM-SR dataset.

    Args:
        fhir_report: FHIR R4 DiagnosticReport dict, OR Bundle containing Observations.
        patient_*: patient demographics (required for DICOM).
        study_instance_uid: must match the source imaging study (for PACS linkage).
        series_instance_uid, sop_instance_uid: generated if None.
        accession_number: links SR to the source study in HIS/RIS.

    Returns:
        FileDataset configured for Comprehensive SR (1.2.840.10008.5.1.4.1.1.88.33)
    """
    now = datetime.now(timezone.utc)
    study_uid = study_instance_uid or generate_uid()
    series_uid = series_instance_uid or generate_uid()
    sop_uid = sop_instance_uid or generate_uid()
    study_date_str = study_date or now.strftime("%Y%m%d")
    study_time_str = now.strftime("%H%M%S")

    # File meta
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = COMPREHENSIVE_SR_SOP_CLASS
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.ImplementationVersionName = "cardio-echo-suite-1"

    ds = FileDataset("/tmp/sr.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Patient module
    ds.PatientID = patient_id
    ds.PatientName = patient_name
    if patient_birth_date:
        ds.PatientBirthDate = patient_birth_date
    if patient_sex:
        ds.PatientSex = patient_sex

    # Study module
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.SOPClassUID = COMPREHENSIVE_SR_SOP_CLASS
    ds.SOPInstanceUID = sop_uid
    ds.AccessionNumber = accession_number or ""
    ds.StudyDescription = study_description
    ds.StudyDate = study_date_str
    ds.StudyTime = study_time_str
    ds.Modality = "SR"  # Structured Report

    # Equipment module
    ds.Manufacturer = manufacturer
    ds.ManufacturerModelName = manufacturer_model_name
    ds.SoftwareVersions = software_versions
    ds.SeriesDate = study_date_str
    ds.SeriesTime = study_time_str
    ds.SeriesNumber = 1
    ds.InstanceNumber = 1

    # SR Document Content module
    # The root is a CONTAINER with concept name "Echocardiography Study" or similar
    ds.ContentDate = study_date_str
    ds.ContentTime = study_time_str

    # Build the SR content tree
    content_sequence: List[Dataset] = []

    # If it's a DiagnosticReport, extract conclusion + contained observations
    if fhir_report.get("resourceType") == "DiagnosticReport":
        # Conclusion as a text content
        if fhir_report.get("conclusion"):
            content_sequence.append(_make_text_content(
                fhir_report["conclusion"],
                relationship="CONTAINS",
            ))

        # Per contained Observation
        for obs in fhir_report.get("contained", []):
            if obs.get("resourceType") != "Observation":
                continue
            content_sequence.extend(_observation_to_content_items(obs))

    # If it's a Bundle, iterate entries
    elif fhir_report.get("resourceType") == "Bundle":
        for entry in fhir_report.get("entry", []):
            obs = entry.get("resource", {})
            if obs.get("resourceType") != "Observation":
                continue
            content_sequence.extend(_observation_to_content_items(obs))

    # Single Observation
    elif fhir_report.get("resourceType") == "Observation":
        content_sequence.extend(_observation_to_content_items(fhir_report))

    # Wrap in root CONTAINER
    root_concept = _make_code_sequence(
        "26520-0", "http://loinc.org", "Echocardiography study"
    )
    ds.ConceptNameCodeSequence = [root_concept]
    ds.ContentTemplateSequence = [_build_template_sequence()]
    ds.ContentSequence = content_sequence

    # Document general module
    ds.CompletionFlag = "COMPLETE"
    ds.VerificationFlag = "UNVERIFIED"  # AI pre-read; flips to VERIFIED after sign-off
    ds.ContentType = "CONTAINER"

    # Patient ID root concept
    instance_creator = Dataset()
    instance_creator.RelationshipType = "HAS CONCEPT MOD"
    instance_creator.ValueType = "CODE"
    instance_creator.ConceptNameCodeSequence = [_make_code_sequence(
        "113050", "DCM", "AI pre-read"
    )]
    instance_creator.ConceptCodeSequence = [_make_code_sequence(
        "1", "http://snomed.info/sct", "AI-generated"
    )]

    return ds


def _observation_to_content_items(obs: Dict[str, Any]) -> List[Dataset]:
    """Convert a FHIR Observation to a list of DICOM-SR content items."""
    items: List[Dataset] = []

    # Get the observation code (LOINC or SNOMED)
    code_coding = obs.get("code", {}).get("coding", [{}])[0]
    obs_code = code_coding.get("code", "")
    obs_system = code_coding.get("system", "")
    obs_meaning = code_coding.get("display", "")

    # Quantitative observation
    if "valueQuantity" in obs:
        vq = obs["valueQuantity"]
        items.append(_make_num_content(
            value=float(vq.get("value", 0.0)),
            unit=vq.get("unit", ""),
            code=obs_code,
            system=obs_system,
            meaning=obs_meaning,
        ))

    # Codeable concept observation (e.g. arrhythmia)
    elif "valueCodeableConcept" in obs:
        vcc = obs["valueCodeableConcept"]
        for coding in vcc.get("coding", []):
            items.append(_make_code_content(
                code=coding.get("code", ""),
                system=coding.get("system", ""),
                meaning=coding.get("display", ""),
                name_code=obs_code,
                name_system=obs_system,
                name_meaning=obs_meaning,
            ))

    # String observation (e.g. draft report text)
    elif "valueString" in obs:
        items.append(_make_text_content(obs["valueString"]))

    # Observation with components (e.g. HRV with multiple metrics)
    for comp in obs.get("component", []):
        comp_code_coding = comp.get("code", {}).get("coding", [{}])[0]
        if "valueQuantity" in comp:
            vq = comp["valueQuantity"]
            items.append(_make_num_content(
                value=float(vq.get("value", 0.0)),
                unit=vq.get("unit", ""),
                code=comp_code_coding.get("code", ""),
                system=comp_code_coding.get("system", ""),
                meaning=comp_code_coding.get("display", ""),
            ))

    return items


def _build_template_sequence() -> Dataset:
    """Build the ContentTemplateSequence for the SR root."""
    tmpl = Dataset()
    tmpl.TemplateIdentifier = "1"
    tmpl.MappingResourceUID = "1.2.840.10008.2.16.4"
    tmpl.TemplateResource = "DCMR"
    return tmpl
