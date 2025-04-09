import logging

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def molstats(smiles_batch):
    """
    Calculates various molecular statistics for a batch of SMILES strings.

    Args:
        smiles_batch (list): A list of strings, where each string is expected
                             to be a SMILES representation of a molecule.

    Returns:
        dict: A dictionary containing aggregated statistics over the valid
              molecules in the batch. Returns None if the input batch is empty.
              Includes:
              - 'total_molecules': Total number of SMILES strings processed.
              - 'valid_molecules': Number of valid SMILES strings.
              - 'validity_percent': Percentage of valid SMILES strings.
              - 'qed_stats': Dict with 'mean', 'std', 'min', 'max' for QED.
              - 'mw_stats': Dict with 'mean', 'std', 'min', 'max' for MolWt.
              - 'logp_stats': Dict with 'mean', 'std', 'min', 'max' for LogP.
              - 'tpsa_stats': Dict with 'mean', 'std', 'min', 'max' for TPSA.
              - 'hba_stats': Dict with 'mean', 'std', 'min', 'max' for H Bond Acceptors.
              - 'hbd_stats': Dict with 'mean', 'std', 'min', 'max' for H Bond Donors.
              - 'rotatable_bonds_stats': Dict with 'mean', 'std', 'min', 'max'.
              - 'num_rings_stats': Dict with 'mean', 'std', 'min', 'max'.
              - 'fraction_csp3_stats': Dict with 'mean', 'std', 'min', 'max'.
              - 'lipinski_violations_stats': Dict with 'mean', 'std', 'min', 'max' for number of violations.
              - 'lipinski_rule_of_5_passed_percent (<=1 violation)': Percentage passing Ro5 (0 or 1 violation).
              - 'lipinski_strict_rule_of_5_passed_percent (0 violations)': Percentage strictly passing Ro5 (0 violations).
              Returns stats calculated only on valid molecules. If no molecules are
              valid, stats fields will contain NaNs or zeros where appropriate.
    """
    if not smiles_batch:
        logging.warning("Input SMILES batch is empty.")
        return None

    total_count = len(smiles_batch)
    valid_molecules = []
    processed_properties = {
        "qed": [],
        "mw": [],
        "logp": [],
        "tpsa": [],
        "hba": [],
        "hbd": [],
        "rotatable_bonds": [],
        "num_rings": [],
        "fraction_csp3": [],
        "lipinski_violations": [],
    }

    logging.info(f"Processing batch of {total_count} SMILES...")
    for smi in smiles_batch:
        if not smi or not isinstance(smi, str):
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            try:
                # Sanitize mol just in case, though MolFromSmiles usually handles basics
                Chem.SanitizeMol(mol)
                valid_molecules.append(mol)

                # Calculate properties
                processed_properties["qed"].append(QED.qed(mol))
                processed_properties["mw"].append(Descriptors.MolWt(mol))
                processed_properties["logp"].append(Crippen.MolLogP(mol))
                processed_properties["tpsa"].append(Descriptors.TPSA(mol))
                processed_properties["hba"].append(Lipinski.NumHAcceptors(mol))
                processed_properties["hbd"].append(Lipinski.NumHDonors(mol))
                processed_properties["rotatable_bonds"].append(
                    Lipinski.NumRotatableBonds(mol)
                )
                processed_properties["num_rings"].append(Lipinski.RingCount(mol))
                processed_properties["fraction_csp3"].append(Lipinski.FractionCSP3(mol))

                # Lipinski Rule of 5 Violations
                violations = 0
                if Descriptors.MolWt(mol) > 500:
                    violations += 1
                if Crippen.MolLogP(mol) > 5:
                    violations += 1
                if Lipinski.NumHDonors(mol) > 5:
                    violations += 1
                if Lipinski.NumHAcceptors(mol) > 10:
                    violations += 1
                processed_properties["lipinski_violations"].append(violations)

            except Exception as e:
                logging.error(f"Error calculating properties for SMILES '{smi}': {e}")
                pass

    valid_count = len(processed_properties["qed"])

    results = {
        "total_molecules": total_count,
        "valid_molecules": valid_count,
        "validity_percent": (valid_count / total_count) * 100 if total_count > 0 else 0,
    }

    if valid_count == 0:
        logging.warning("No valid molecules found in the batch.")
        # Fill stats with NaN or appropriate defaults
        nan_stats = {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
        for key in processed_properties:
            results[f"{key}_stats"] = nan_stats.copy()
        results["lipinski_rule_of_5_passed_percent (<=1 violation)"] = 0.0
        results["lipinski_strict_rule_of_5_passed_percent (0 violations)"] = 0.0
        return results

    # Calculate statistics for each property using NumPy
    for key, values in processed_properties.items():
        values_np = np.array(values)
        results[f"{key}_stats"] = {
            "mean": np.mean(values_np),
            "std": np.std(values_np),
            "min": np.min(values_np),
            "max": np.max(values_np),
        }

    # Calculate Lipinski pass rates
    violations_np = np.array(processed_properties["lipinski_violations"])
    passed_lipinski_le1 = np.sum(violations_np <= 1)
    passed_lipinski_strict = np.sum(violations_np == 0)

    results["lipinski_rule_of_5_passed_percent (<=1 violation)"] = (
        passed_lipinski_le1 / valid_count
    ) * 100
    results["lipinski_strict_rule_of_5_passed_percent (0 violations)"] = (
        passed_lipinski_strict / valid_count
    ) * 100

    logging.info(f"Processed batch. Validity: {results['validity_percent']:.2f}%")

    return results


def print_molstats(stats):
    if stats is None:
        print("No statistics available.")
        return

    print("\n--- Molecular Statistics ---")
    print(f"Total Molecules Processed: {stats['total_molecules']}")
    print(f"Valid Molecules Found:     {stats['valid_molecules']}")
    print(f"Validity Percentage:       {stats['validity_percent']:.2f}%")
    print("-" * 28)

    def print_metric(name, data_key):
        if f"{data_key}_stats" in stats:
            s = stats[f"{data_key}_stats"]
            if not np.isnan(s["mean"]):
                print(f"{name}:")
                print(f"  Mean: {s['mean']:.3f}")
                print(f"  Std:  {s['std']:.3f}")
                print(f"  Min:  {s['min']:.3f}")
                print(f"  Max:  {s['max']:.3f}")
            else:
                print(f"{name}: N/A (No valid molecules)")
        else:
            print(f"{name}: Not Calculated")

    print_metric("QED", "qed")
    print_metric("Molecular Weight (MW)", "mw")
    print_metric("LogP (Crippen)", "logp")
    print_metric("Topological Polar Surface Area (TPSA)", "tpsa")
    print_metric("H-Bond Acceptors (HBA)", "hba")
    print_metric("H-Bond Donors (HBD)", "hbd")
    print_metric("Rotatable Bonds", "rotatable_bonds")
    print_metric("Number of Rings", "num_rings")
    print_metric("Fraction Csp3", "fraction_csp3")
    print_metric("Lipinski Violations", "lipinski_violations")

    print("-" * 28)
    print("Drug-Likeness (Lipinski Rule of 5):")
    print(
        f"  % Passing (<=1 Violation): {stats.get('lipinski_rule_of_5_passed_percent (<=1 violation)', 0.0):.2f}%"
    )
    print(
        f"  % Strictly Passing (0 Violations): {stats.get('lipinski_strict_rule_of_5_passed_percent (0 violations)', 0.0):.2f}%"
    )
    print("--- End Statistics ---\n")
