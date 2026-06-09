from __future__ import annotations

from pathlib import Path
import subprocess


class DWGConverter:
    def convert(self, input_path: Path, output_dir: Path) -> Path:
        raise NotImplementedError


class ODAFileConverter(DWGConverter):
    def __init__(self, exe_path: Path, version: str = "ACAD2018") -> None:
        self.exe_path = exe_path
        self.version = version

    def convert(self, input_path: Path, output_dir: Path) -> Path:
        input_dir = input_path.parent.resolve()
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(self.exe_path),
            str(input_dir),
            str(output_dir),
            self.version,
            "DXF",
            "0",  # recursive
            "1",  # audit
        ]
        subprocess.run(cmd, check=True)

        return output_dir / input_path.with_suffix(".dxf").name


class LibreDWGConverter(DWGConverter):
    def __init__(self, exe_path: Path) -> None:
        self.exe_path = exe_path

    def convert(self, input_path: Path, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / input_path.with_suffix(".dxf").name

        cmd = [
            str(self.exe_path),
            str(input_path),
            "-o",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)
        return output_path
