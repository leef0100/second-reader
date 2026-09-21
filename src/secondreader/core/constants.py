"""Project-wide constants. Import these; never retype the research-use label."""

RESEARCH_USE_LABEL = "For research use only. Not for diagnostic use."

# Manufacturer / software identity written into every DICOM object we emit.
MANUFACTURER = "Second Reader (research prototype)"
SOFTWARE_VERSIONS = "secondreader 0.1.0"

# Standard finding identifiers are per-modality; these are the platform-level
# result states shared by every plugin.
GATE_OK = "ok"
GATE_UNABLE_TO_ASSESS = "unable_to_assess"
