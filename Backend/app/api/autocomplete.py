from fastapi import APIRouter, Query
from app.services.training_data_service import autocomplete_drugs, autocomplete_proteins

router = APIRouter()

@router.get("/autocomplete/drug")
def drug_autocomplete(q: str = Query("")):
    return autocomplete_drugs(q)

@router.get("/autocomplete/protein")
def protein_autocomplete(q: str = Query("")):
    return autocomplete_proteins(q)