from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from assemble_paper_mechanism_package import figure_catalog, results_evidence_matrix  # noqa: E402


class PaperMechanismPackageTests(unittest.TestCase):
    def test_catalog_has_complete_main_and_supplement_claim_boundaries(self) -> None:
        catalog = figure_catalog()
        main = catalog[catalog["placement"] == "main"]
        supplement = catalog[catalog["placement"] == "supplement"]

        self.assertEqual(main["figure_id"].tolist(), [f"Fig. {index}" for index in range(1, 7)])
        self.assertEqual(len(supplement), 6)
        self.assertTrue(main["file"].str.contains("paper_contact_evolution_framework.png").any())
        self.assertTrue(catalog["allowed_claim"].str.strip().ne("").all())
        self.assertTrue(catalog["prohibited_claim"].str.strip().ne("").all())
        self.assertFalse(main["uncertainty_representation"].str.contains("confidence interval", case=False).any())

        evidence = results_evidence_matrix()
        self.assertEqual(evidence["section"].tolist(), [f"3.{index}" for index in range(1, 8)])
        self.assertTrue(evidence["required_restriction"].str.strip().ne("").all())


if __name__ == "__main__":
    unittest.main()
