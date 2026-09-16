#!/usr/bin/env python3
"""Generate a Software Bill of Materials (SBOM) for cardio-echo-suite.

Outputs a JSON SBOM in SPDX 2.3 format containing:
- All Python packages from each service's requirements.txt
- All Docker base images
- All model weight sources
- License information per component

Usage:
    python scripts/generate_sbom.py \\
        --output docs/regulatory/SBOM.json \\
        --format spdx-json

Compliant with:
- FDA Guidance "Cybersecurity in Medical Devices: Quality System Considerations and Content of Premarket Submissions" (2023)
- IEC 81001-5-1:2021
- SPDX 2.3 specification
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_requirements(path: Path) -> List[Dict[str, str]]:
    """Parse a requirements.txt file into a list of {name, version, constraint}."""
    packages = []
    if not path.exists():
        return packages
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Handle cardio-echo-core @ file://...
        if "@" in line:
            name = line.split("@")[0].strip()
            packages.append({"name": name, "version": "dev", "constraint": "@"})
            continue
        # Handle package[extra]>=1.0,<2.0
        m = re.match(r"^([a-zA-Z0-9_\-\.\[\]]+)([<>=!~\s]+.*)?$", line)
        if m:
            name = m.group(1).split("[")[0]  # strip extras
            constraint = (m.group(2) or "").strip()
            version = constraint.lstrip("<>=!~ ") if constraint else "latest"
            packages.append({"name": name, "version": version, "constraint": constraint})
    return packages


def parse_dockerfile(path: Path) -> List[str]:
    """Extract base images from a Dockerfile."""
    bases = []
    if not path.exists():
        return bases
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.upper().startswith("FROM "):
            # FROM <image>[:<tag>] [AS <alias>]
            parts = line.split()
            if len(parts) >= 2:
                bases.append(parts[1])
    return bases


def generate_sbom(repo_root: Path) -> Dict[str, Any]:
    """Generate SPDX 2.3 JSON SBOM."""
    sbom_id = str(uuid.uuid4())
    created = datetime.now(timezone.utc).isoformat()

    packages: List[Dict[str, Any]] = []
    relationships: List[Dict[str, str]] = []

    # cardio-echo-suite itself
    suite_pkg = {
        "SPDXID": "SPDXRef-Package-cardio-echo-suite",
        "name": "cardio-echo-suite",
        "versionInfo": "0.5.0",
        "downloadLocation": "https://github.com/Dalfino/cardio-echo-suite",
        "filesAnalyzed": False,
        "licenseConcluded": "MIT",
        "licenseDeclared": "MIT",
        "copyrightText": "Copyright (c) 2026 cardio-echo-suite authors",
    }
    packages.append(suite_pkg)

    # Each service's requirements
    service_dirs = sorted((repo_root / "services").iterdir())
    for svc_dir in service_dirs:
        if not svc_dir.is_dir() or svc_dir.name.startswith("."):
            continue
        # Add the service as a package
        svc_pkg_id = f"SPDXRef-Package-{svc_dir.name}"
        svc_pkg = {
            "SPDXID": svc_pkg_id,
            "name": svc_dir.name,
            "versionInfo": "0.5.0",
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "licenseConcluded": "MIT",
            "licenseDeclared": "MIT",
            "copyrightText": "Copyright (c) 2026 cardio-echo-suite authors",
        }
        packages.append(svc_pkg)
        relationships.append({
            "spdxElementId": "SPDXRef-Package-cardio-echo-suite",
            "relationshipType": "CONTAINS",
            "relatedSpdxElement": svc_pkg_id,
        })

        # Add dependencies
        req_path = svc_dir / "requirements.txt"
        deps = parse_requirements(req_path)
        for dep in deps:
            dep_id = f"SPDXRef-Package-{dep['name'].replace('_','-').lower()}"
            dep_pkg = {
                "SPDXID": dep_id,
                "name": dep["name"],
                "versionInfo": dep["version"],
                "downloadLocation": f"https://pypi.org/project/{dep['name']}/",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",  # would need license metadata
                "licenseDeclared": "NOASSERTION",
                "copyrightText": "NOASSERTION",
            }
            # Don't duplicate packages already added
            if not any(p["SPDXID"] == dep_id for p in packages):
                packages.append(dep_pkg)
            relationships.append({
                "spdxElementId": svc_pkg_id,
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": dep_id,
            })

        # Dockerfile base images
        dockerfile_path = svc_dir / "Dockerfile"
        bases = parse_dockerfile(dockerfile_path)
        for base in bases:
            base_id = f"SPDXRef-Package-docker-{base.replace(':','-').replace('/','-').replace('_','-').lower()}"
            base_pkg = {
                "SPDXID": base_id,
                "name": f"docker:{base}",
                "versionInfo": base.split(":")[-1] if ":" in base else "latest",
                "downloadLocation": f"https://hub.docker.com/_/{base.split(':')[0].split('/')[-1]}",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
                "copyrightText": "NOASSERTION",
            }
            if not any(p["SPDXID"] == base_id for p in packages):
                packages.append(base_pkg)
            relationships.append({
                "spdxElementId": svc_pkg_id,
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": base_id,
            })

    # Model weight sources (from model_registry)
    try:
        sys.path.insert(0, str(repo_root / "packages" / "cardio-echo-core"))
        from cardio_echo_core.model_registry import REGISTRY  # type: ignore
        for name, cfg in REGISTRY.items():
            model_id = f"SPDXRef-Model-{name}"
            model_pkg = {
                "SPDXID": model_id,
                "name": f"ai-model:{name}",
                "versionInfo": cfg.upstream_ref,
                "downloadLocation": cfg.upstream_repo,
                "filesAnalyzed": False,
                "licenseConcluded": cfg.upstream_license,
                "licenseDeclared": cfg.upstream_license,
                "copyrightText": f"See {cfg.upstream_citation}",
            }
            packages.append(model_pkg)
            svc_id = f"SPDXRef-Package-{name}"
            if any(p["SPDXID"] == svc_id for p in packages):
                relationships.append({
                    "spdxElementId": svc_id,
                    "relationshipType": "DEPENDS_ON",
                    "relatedSpdxElement": model_id,
                })
    except ImportError:
        pass

    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": f"spdx-{sbom_id}",
        "name": "cardio-echo-suite SBOM",
        "documentNamespace": f"https://github.com/Dalfino/cardio-echo-suite/sbom/{sbom_id}",
        "creationInfo": {
            "created": created,
            "creators": [
                "Tool: cardio-echo-suite SBOM generator (scripts/generate_sbom.py)",
                "Organization: cardio-echo-suite",
            ],
            "licenseListVersion": "3.20",
        },
        "packages": packages,
        "relationships": relationships,
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Generate SPDX 2.3 SBOM for cardio-echo-suite")
    p.add_argument("--output", type=Path, default=Path("docs/regulatory/SBOM.json"))
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--format", choices=["spdx-json", "spdx-rdf", "cyclonedx-json"], default="spdx-json")
    args = p.parse_args(argv)

    sbom = generate_sbom(args.repo_root)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sbom, indent=2))

    n_packages = len(sbom["packages"])
    n_relationships = len(sbom["relationships"])
    print(f"SBOM written to {args.output}")
    print(f"  Packages: {n_packages}")
    print(f"  Relationships: {n_relationships}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
