from fastapi import APIRouter, Query
from typing import Optional
from app.services.training_data_service import autocomplete_drugs, autocomplete_proteins, get_random_drug_sample
from app.services.drug_descriptor_service import get_available_filters, filter_drugs, count_drugs

router = APIRouter()

@router.get("/autocomplete/drug")
def drug_autocomplete(q: str = Query("")):
    return autocomplete_drugs(q)

@router.get("/autocomplete/protein")
def protein_autocomplete(q: str = Query("")):
    return autocomplete_proteins(q)

@router.get("/autocomplete/drug-sample")
def drug_sample(n: int = Query(20, ge=1, le=50)):
    """Return a random sample of drugs for quick screening."""
    return get_random_drug_sample(n)

@router.get("/drug-filters")
def drug_filters():
    """Return available filter definitions for the frontend."""
    return get_available_filters()


def _filter_params(
    lipinski: bool = Query(False),
    mw_min: Optional[float] = Query(None),
    mw_max: Optional[float] = Query(None),
    rings_min: Optional[int] = Query(None),
    rings_max: Optional[int] = Query(None),
    known_binders_target: Optional[str] = Query(None),
):
    return dict(
        lipinski=lipinski,
        mw_min=mw_min,
        mw_max=mw_max,
        rings_min=rings_min,
        rings_max=rings_max,
        known_binders_target=known_binders_target,
    )


@router.get("/drug-filter")
def drug_filter(
    lipinski: bool = Query(False),
    mw_min: Optional[float] = Query(None),
    mw_max: Optional[float] = Query(None),
    rings_min: Optional[int] = Query(None),
    rings_max: Optional[int] = Query(None),
    known_binders_target: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
):
    """Return drugs matching the given physicochemical / activity filters."""
    return filter_drugs(
        lipinski=lipinski,
        mw_min=mw_min,
        mw_max=mw_max,
        rings_min=rings_min,
        rings_max=rings_max,
        known_binders_target=known_binders_target,
        limit=limit,
    )


@router.get("/drug-filter/count")
def drug_filter_count(
    lipinski: bool = Query(False),
    mw_min: Optional[float] = Query(None),
    mw_max: Optional[float] = Query(None),
    rings_min: Optional[int] = Query(None),
    rings_max: Optional[int] = Query(None),
    known_binders_target: Optional[str] = Query(None),
):
    """Return the count of compounds matching the given filters."""
    return {
        "count": count_drugs(
            lipinski=lipinski,
            mw_min=mw_min,
            mw_max=mw_max,
            rings_min=rings_min,
            rings_max=rings_max,
            known_binders_target=known_binders_target,
        )
    }