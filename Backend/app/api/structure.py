from fastapi import APIRouter, Query
from fastapi.responses import Response
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

router = APIRouter()


@router.get("/structure/svg")
def structure_svg(
    smiles: str = Query(...),
    w: int = Query(200, ge=50, le=500),
    h: int = Query(150, ge=50, le=500),
):
    """Return an SVG depiction of a molecule from its SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return Response(
            content='<svg xmlns="http://www.w3.org/2000/svg" width="200" height="150">'
            '<text x="100" y="75" text-anchor="middle" fill="#999" font-size="12">'
            "Invalid SMILES</text></svg>",
            media_type="image/svg+xml",
        )

    drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
    drawer.drawOptions().clearBackground = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    return Response(content=svg, media_type="image/svg+xml")
