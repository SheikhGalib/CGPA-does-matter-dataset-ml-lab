# Build instructions

Run from this `report` directory in PowerShell:

```powershell
xelatex -interaction=nonstopmode -halt-on-error main.tex
biber main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

After the final XeLaTeX run, create the named submission copy with:

```powershell
Copy-Item main.pdf ..\output\pdf\What_Shapes_Student_CGPA_Report.pdf -Force
```

The repository also includes that named PDF under `report/` for direct download.

The report intentionally retains TODO boxes for ethics, confirmed contributor roles, acknowledgements, funding, competing interests, and an optional public repository URL until the team confirms those statements.
