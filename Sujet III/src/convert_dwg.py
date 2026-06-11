from __future__ import annotations

from pathlib import Path
import subprocess


def convert_dwg_to_dxf(
    oda_exe: Path,
    input_dir: Path,
    output_dir: Path,
    version: str = "ACAD2018",
    recursive: bool = False,
    audit: bool = True,
) -> None:
    """
    Convertit tous les DWG/DXF trouvés dans input_dir vers DXF dans output_dir
    en appelant Teigha / ODA File Converter.

    Signature volontairement compatible avec pipeline.py:
        convert_dwg_to_dxf(
            oda_exe=args.oda_exe,
            input_dir=dwg_input_dir,
            output_dir=dxf_output_dir,
        )

    Paramètres
    ----------
    oda_exe : Path
        Chemin vers l'exécutable ODAFileConverter.exe / TeighaFileConverter.exe
    input_dir : Path
        Dossier source contenant les DWG
    output_dir : Path
        Dossier cible
    version : str
        Version de sortie demandée par le convertisseur (ex: ACAD2018)
    recursive : bool
        Parcours récursif des sous-dossiers
    audit : bool
        Audit/réparation si supporté par le convertisseur
    """
    oda_exe = Path(oda_exe).resolve()
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if not oda_exe.exists():
        raise FileNotFoundError(f"Exécutable convertisseur introuvable : {oda_exe}")

    if not input_dir.exists():
        raise FileNotFoundError(f"Dossier d'entrée introuvable : {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    recurse_flag = "1" if recursive else "0"
    audit_flag = "1" if audit else "0"

    # Format de ligne de commande attendu par ODA/Teigha File Converter :
    #   <in_folder> <out_folder> <version> <type> <recursive> <audit>
    cmd = [
        str(oda_exe),
        str(input_dir),
        str(output_dir),
        version,
        "DXF",
        recurse_flag,
        audit_flag,
    ]

    print(f"[convert_dwg_to_dxf] cmd={' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        err = result.stderr.strip() if result.stderr else "Erreur inconnue du convertisseur."
        raise RuntimeError(
            "Échec de la conversion DWG -> DXF.\n"
            f"Code retour: {result.returncode}\n"
            f"stderr: {err}"
        )
