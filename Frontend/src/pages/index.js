import * as React from "react"
import Layout from "../components/layout"
import Seo from "../components/seo"
import * as styles from "../components/index.module.css"

const API_BASE = process.env.GATSBY_API_BASE_URL || "http://127.0.0.1:8001"

const IndexPage = () => {
  // --- mode toggle ---
  const [mode, setMode] = React.useState("single") // "single" | "screen"

  // --- single predict state ---
  const [drug, setDrug] = React.useState("")
  const [drugSmiles, setDrugSmiles] = React.useState("")
  const [protein, setProtein] = React.useState("")
  const [loading, setLoading] = React.useState(false)
  const [error, setError] = React.useState("")
  const [result, setResult] = React.useState(null)
  const [history, setHistory] = React.useState([])

  const [drugSuggestions, setDrugSuggestions] = React.useState([])
  const [proteinSuggestions, setProteinSuggestions] = React.useState([])

  // --- screening state ---
  const [screenDrugs, setScreenDrugs] = React.useState([]) // [{name, smiles}]
  const [screenProtein, setScreenProtein] = React.useState("")
  const [screenProteinSuggestions, setScreenProteinSuggestions] = React.useState([])
  const [screenDrugQuery, setScreenDrugQuery] = React.useState("")
  const [screenDrugSuggestions, setScreenDrugSuggestions] = React.useState([])
  const [screenLoading, setScreenLoading] = React.useState(false)
  const [screenError, setScreenError] = React.useState("")
  const [screenResults, setScreenResults] = React.useState(null)

  // --- results table state ---
  const [sortCol, setSortCol] = React.useState("rank")
  const [sortAsc, setSortAsc] = React.useState(true)
  const [hoverSmiles, setHoverSmiles] = React.useState(null)
  const [hoverPos, setHoverPos] = React.useState({ x: 0, y: 0 })

  // --- compound library filters ---
  const [filterLipinski, setFilterLipinski] = React.useState(false)
  const [filterMwPreset, setFilterMwPreset] = React.useState("all")
  const [filterRingsPreset, setFilterRingsPreset] = React.useState("all")
  const [filterBinderTarget, setFilterBinderTarget] = React.useState("")
  const [filterBinderSuggestions, setFilterBinderSuggestions] = React.useState([])
  const [filterLoading, setFilterLoading] = React.useState(false)
  const [filterCount, setFilterCount] = React.useState(null)
  const [filterCountLoading, setFilterCountLoading] = React.useState(false)

  const MW_PRESETS = [
    { value: "all", label: "All", min: null, max: null },
    { value: "fragment", label: "Fragment-like (100\u2013300 Da)", min: 100, max: 300 },
    { value: "lead", label: "Lead-like (250\u2013350 Da)", min: 250, max: 350 },
    { value: "drug", label: "Drug-like (300\u2013500 Da)", min: 300, max: 500 },
    { value: "macro", label: "Macrocyclic (500\u20131000 Da)", min: 500, max: 1000 },
    { value: "beyond", label: "Beyond Ro5 (>500 Da)", min: 500, max: null },
  ]

  const RINGS_PRESETS = [
    { value: "all", label: "All", min: null, max: null },
    { value: "acyclic", label: "Acyclic (0)", min: 0, max: 0 },
    { value: "simple", label: "Simple (1\u20132)", min: 1, max: 2 },
    { value: "moderate", label: "Moderate (3\u20134)", min: 3, max: 4 },
    { value: "complex", label: "Complex (5+)", min: 5, max: null },
  ]

  const getFilterParams = () => {
    const mw = MW_PRESETS.find((p) => p.value === filterMwPreset) || MW_PRESETS[0]
    const rings = RINGS_PRESETS.find((p) => p.value === filterRingsPreset) || RINGS_PRESETS[0]
    const params = new URLSearchParams()
    if (filterLipinski) params.set("lipinski", "true")
    if (mw.min != null) params.set("mw_min", String(mw.min))
    if (mw.max != null) params.set("mw_max", String(mw.max))
    if (rings.min != null) params.set("rings_min", String(rings.min))
    if (rings.max != null) params.set("rings_max", String(rings.max))
    if (filterBinderTarget) params.set("known_binders_target", filterBinderTarget)
    return params
  }

  const hasActiveFilters = filterLipinski || filterMwPreset !== "all" || filterRingsPreset !== "all" || filterBinderTarget !== ""

  const resetFilters = () => {
    setFilterLipinski(false)
    setFilterMwPreset("all")
    setFilterRingsPreset("all")
    setFilterBinderTarget("")
    setFilterBinderSuggestions([])
    setFilterCount(null)
  }

  // Helper to highlight matched query safely
  const highlightMatch = (text = "", query = "") => {
    const idx = text.toLowerCase().indexOf(query.toLowerCase())
    if (idx === -1) return text
    return (
      text.substring(0, idx) +
      "<mark>" +
      text.substring(idx, idx + query.length) +
      "</mark>" +
      text.substring(idx + query.length)
    )
  }

  // Debounce hook
  const useDebounce = (callback, delay) => {
    const timeoutRef = React.useRef(null)
    return (...args) => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
      timeoutRef.current = setTimeout(() => callback(...args), delay)
    }
  }

  const updateDrugSuggestions = useDebounce(async (value) => {
    if (!value) { setDrugSuggestions([]); return }
    try {
      const res = await fetch(`${API_BASE}/autocomplete/drug?q=${encodeURIComponent(value)}`)
      const raw = await res.json()
      const data = Array.isArray(raw) ? raw : []
      setDrugSuggestions(data.length > 0 ? data : [{ name: "No results found", smiles: "" }])
    } catch (err) {
      console.error("Drug autocomplete failed:", err)
      setDrugSuggestions([{ name: "No results found", smiles: "" }])
    }
  }, 300)

  const updateProteinSuggestions = useDebounce(async (value) => {
    if (!value) { setProteinSuggestions([]); return }
    try {
      const res = await fetch(`${API_BASE}/autocomplete/protein?q=${encodeURIComponent(value)}`)
      const raw = await res.json()
      const data = Array.isArray(raw) ? raw : []
      setProteinSuggestions(data.length > 0 ? data : ["No results found"])
    } catch (err) {
      console.error("Protein autocomplete failed:", err)
      setProteinSuggestions(["No results found"])
    }
  }, 300)

  // Screening autocomplete
  const updateScreenDrugSuggestions = useDebounce(async (value) => {
    if (!value) { setScreenDrugSuggestions([]); return }
    try {
      const res = await fetch(`${API_BASE}/autocomplete/drug?q=${encodeURIComponent(value)}`)
      const raw = await res.json()
      const data = Array.isArray(raw) ? raw : []
      setScreenDrugSuggestions(data.length > 0 ? data : [{ name: "No results found", smiles: "" }])
    } catch (err) {
      setScreenDrugSuggestions([{ name: "No results found", smiles: "" }])
    }
  }, 300)

  const updateScreenProteinSuggestions = useDebounce(async (value) => {
    if (!value) { setScreenProteinSuggestions([]); return }
    try {
      const res = await fetch(`${API_BASE}/autocomplete/protein?q=${encodeURIComponent(value)}`)
      const raw = await res.json()
      const data = Array.isArray(raw) ? raw : []
      setScreenProteinSuggestions(data.length > 0 ? data : ["No results found"])
    } catch (err) {
      setScreenProteinSuggestions(["No results found"])
    }
  }, 300)

  // Binder target autocomplete (reuses protein autocomplete endpoint)
  const updateFilterBinderSuggestions = useDebounce(async (value) => {
    if (!value) { setFilterBinderSuggestions([]); return }
    try {
      const res = await fetch(`${API_BASE}/autocomplete/protein?q=${encodeURIComponent(value)}`)
      const raw = await res.json()
      const data = Array.isArray(raw) ? raw : []
      setFilterBinderSuggestions(data.length > 0 ? data : ["No results found"])
    } catch (err) {
      setFilterBinderSuggestions(["No results found"])
    }
  }, 300)

  const fetchFilterCount = useDebounce(async () => {
    if (!hasActiveFilters) { setFilterCount(null); return }
    setFilterCountLoading(true)
    try {
      const params = getFilterParams()
      const res = await fetch(`${API_BASE}/drug-filter/count?${params.toString()}`)
      const data = await res.json()
      setFilterCount(data.count)
    } catch (err) {
      console.error("Count failed:", err)
    } finally {
      setFilterCountLoading(false)
    }
  }, 400)

  // Trigger live count whenever filters change
  React.useEffect(() => {
    fetchFilterCount()
  }, [filterLipinski, filterMwPreset, filterRingsPreset, filterBinderTarget])

  const loadFilteredDrugs = async () => {
    const params = getFilterParams()
    params.set("limit", "50")

    setFilterLoading(true)
    try {
      const res = await fetch(`${API_BASE}/drug-filter?${params.toString()}`)
      const data = await res.json()
      if (Array.isArray(data)) {
        const unique = data.filter(
          (d) => !screenDrugs.some((existing) => existing.smiles === d.smiles)
        )
        setScreenDrugs((prev) => [...prev, ...unique])
      }
    } catch (err) {
      console.error("Filter failed:", err)
    } finally {
      setFilterLoading(false)
    }
  }

  const fetchHistory = React.useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/history`)
      if (!res.ok) throw new Error("Failed to load history")
      const data = await res.json()
      setHistory(data)
    } catch (e) {
      console.error("Failed to fetch history:", e)
    }
  }, [])

  React.useEffect(() => { fetchHistory() }, [fetchHistory])

  // --- single predict ---
  const onSubmit = async (e) => {
    e.preventDefault()
    setError("")
    setResult(null)
    const drugToSend = drugSmiles || drug
    if (!drugToSend || !protein) {
      setError("Please enter both a drug and a protein.")
      return
    }
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ drug: drugToSend, drug_name: drug || null, protein }),
      })
      if (!res.ok) {
        const msg = await res.text()
        throw new Error(msg || "Prediction failed")
      }
      const data = await res.json()
      setResult(data)
      fetchHistory()
    } catch (e) {
      setError(e.message || "Something went wrong")
    } finally {
      setLoading(false)
    }
  }

  const onClearHistory = async () => {
    try {
      await fetch(`${API_BASE}/history`, { method: "DELETE" })
      setHistory([])
    } catch (e) {
      console.error("Failed to clear history:", e)
    }
  }

  // --- screening ---
  const addScreenDrug = (d) => {
    if (screenDrugs.some((existing) => existing.smiles === d.smiles)) return
    setScreenDrugs((prev) => [...prev, d])
    setScreenDrugQuery("")
    setScreenDrugSuggestions([])
  }

  const removeScreenDrug = (smiles) => {
    setScreenDrugs((prev) => prev.filter((d) => d.smiles !== smiles))
  }

  const loadRandomSample = async () => {
    try {
      const res = await fetch(`${API_BASE}/autocomplete/drug-sample?n=20`)
      const data = await res.json()
      if (Array.isArray(data)) {
        const unique = data.filter(
          (d) => !screenDrugs.some((existing) => existing.smiles === d.smiles)
        )
        setScreenDrugs((prev) => [...prev, ...unique])
      }
    } catch (err) {
      console.error("Failed to load random sample:", err)
    }
  }

  const onScreen = async (e) => {
    e.preventDefault()
    setScreenError("")
    setScreenResults(null)
    if (screenDrugs.length === 0) {
      setScreenError("Add at least one drug to screen.")
      return
    }
    if (!screenProtein) {
      setScreenError("Select a protein target.")
      return
    }
    setScreenLoading(true)
    try {
      const res = await fetch(`${API_BASE}/screen`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          drugs: screenDrugs.map((d) => ({ smiles: d.smiles, name: d.name || null })),
          protein: screenProtein,
        }),
      })
      if (!res.ok) {
        const msg = await res.text()
        throw new Error(msg || "Screening failed")
      }
      const data = await res.json()
      setScreenResults(data)
    } catch (e) {
      setScreenError(e.message || "Something went wrong")
    } finally {
      setScreenLoading(false)
    }
  }

  // --- results table helpers ---
  const handleSort = (col) => {
    if (sortCol === col) {
      setSortAsc(!sortAsc)
    } else {
      setSortCol(col)
      setSortAsc(col === "rank" || col === "drug_name") // asc for rank/name, desc for scores
    }
  }

  const getSortedHits = () => {
    if (!screenResults) return []
    const hits = [...screenResults.hits]
    hits.sort((a, b) => {
      let av = a[sortCol], bv = b[sortCol]
      // Handle nulls
      if (av == null && bv == null) return 0
      if (av == null) return 1
      if (bv == null) return -1
      // String comparison for names
      if (typeof av === "string") return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av)
      // Boolean - binder first
      if (typeof av === "boolean") { av = av ? 0 : 1; bv = bv ? 0 : 1 }
      return sortAsc ? av - bv : bv - av
    })
    return hits
  }

  const sortIndicator = (col) => {
    if (sortCol !== col) return " \u2195"
    return sortAsc ? " \u2191" : " \u2193"
  }

  const downloadCsv = () => {
    if (!screenResults) return
    const rows = getSortedHits()
    const header = "Rank,Compound,SMILES,MW,LogP,Rings,Score,Prediction"
    const csvRows = rows.map((h, i) => {
      const name = (h.drug_name || "").replace(/"/g, '""')
      const smiles = h.drug.replace(/"/g, '""')
      return [
        i + 1,
        `"${name}"`,
        `"${smiles}"`,
        h.mw != null ? h.mw.toFixed(1) : "",
        h.logp != null ? h.logp.toFixed(2) : "",
        h.rings != null ? h.rings : "",
        h.score.toFixed(4),
        h.binder ? "Binder" : "Non-binder",
      ].join(",")
    })
    const csv = [header, ...csvRows].join("\n")
    const blob = new Blob([csv], { type: "text/csv" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `screening_${screenResults.protein.replace(/[^a-zA-Z0-9]/g, "_")}_${new Date().toISOString().slice(0, 10)}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  const thStyle = (col) => ({
    padding: "8px 10px",
    borderBottom: "2px solid #ddd",
    cursor: "pointer",
    userSelect: "none",
    whiteSpace: "nowrap",
    background: sortCol === col ? "#e8edf5" : "#f5f5f5",
  })

  // --- shared styles ---
  const tabStyle = (active) => ({
    padding: "10px 24px",
    cursor: "pointer",
    border: "1px solid #ccc",
    borderBottom: active ? "2px solid #1a73e8" : "1px solid #ccc",
    background: active ? "#f0f4ff" : "#fff",
    fontWeight: active ? 600 : 400,
    borderRadius: "6px 6px 0 0",
    marginRight: 4,
  })

  const chipStyle = {
    display: "inline-flex",
    alignItems: "center",
    gap: 4,
    background: "#e8f0fe",
    border: "1px solid #a8c7fa",
    borderRadius: 16,
    padding: "4px 10px",
    fontSize: 13,
  }

  return (
    <Layout>
      <div style={{ maxWidth: 800, margin: "0 auto" }}>
        <h1>Drug–Target Interaction (DTI) Checker</h1>
        <p>Predict binding between drugs and protein targets.</p>

        {/* Mode tabs */}
        <div style={{ display: "flex", marginBottom: 0 }}>
          <button style={tabStyle(mode === "single")} onClick={() => setMode("single")}>
            Single Prediction
          </button>
          <button style={tabStyle(mode === "screen")} onClick={() => setMode("screen")}>
            Virtual Screening
          </button>
        </div>
        <div style={{ border: "1px solid #ccc", borderTop: "none", padding: 20, borderRadius: "0 0 8px 8px", marginBottom: 24 }}>

          {/* ===== SINGLE PREDICTION MODE ===== */}
          {mode === "single" && (
            <>
              <form onSubmit={onSubmit} style={{ marginBottom: 16 }}>
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  <label>
                    <div>Drug</div>
                    <input
                      type="text"
                      value={drug}
                      onChange={(e) => {
                        const v = e.target.value
                        setDrug(v)
                        setDrugSmiles("")
                        updateDrugSuggestions(v)
                      }}
                      placeholder="e.g., Imatinib"
                      style={{ width: "100%", padding: 8 }}
                    />
                    {drugSuggestions.length > 0 && drug && (
                      <ul style={{
                        border: "1px solid #888", padding: 0, background: "#fff", marginTop: 4,
                        listStyle: "none", borderRadius: 6, boxShadow: "0px 2px 6px rgba(0,0,0,0.15)",
                        maxHeight: 180, overflowY: "auto"
                      }}>
                        {drugSuggestions.map((d, idx) => {
                          const display = d?.name || d?.smiles || "Unknown"
                          const isNoResult = display === "No results found"
                          return (
                            <li key={idx}
                              style={{ cursor: "pointer", padding: "6px 10px", borderBottom: "1px solid #eee" }}
                              onClick={() => { if (isNoResult) return; setDrug(display); setDrugSmiles(d.smiles || ""); setDrugSuggestions([]) }}
                              onMouseEnter={(e) => (e.currentTarget.style.background = "#f0f4ff")}
                              onMouseLeave={(e) => (e.currentTarget.style.background = "white")}
                              title={d?.smiles ? `SMILES: ${d.smiles}` : ""}
                            >
                              <span dangerouslySetInnerHTML={{ __html: isNoResult ? display : highlightMatch(display, drug) }} />
                            </li>
                          )
                        })}
                      </ul>
                    )}
                    {drugSmiles && (
                      <small style={{ color: "#666", display: "block", marginTop: 6 }}>
                        Using selected SMILES for prediction.
                      </small>
                    )}
                  </label>

                  <label>
                    <div>Protein</div>
                    <input
                      type="text"
                      value={protein}
                      onChange={(e) => { setProtein(e.target.value); updateProteinSuggestions(e.target.value) }}
                      placeholder="e.g., Cytochrome P450 3A4"
                      style={{ width: "100%", padding: 8 }}
                    />
                    {proteinSuggestions.length > 0 && protein && (
                      <ul style={{
                        border: "1px solid #888", padding: 0, background: "#fff", marginTop: 4,
                        listStyle: "none", borderRadius: 6, boxShadow: "0px 2px 6px rgba(0,0,0,0.15)",
                        maxHeight: 180, overflowY: "auto"
                      }}>
                        {proteinSuggestions.map((p, idx) => (
                          <li key={idx}
                            style={{ cursor: "pointer", padding: "6px 10px", borderBottom: "1px solid #eee" }}
                            onClick={() => { if (p !== "No results found") { setProtein(p); setProteinSuggestions([]) } }}
                            onMouseEnter={(e) => e.currentTarget.style.background = "#f0f4ff"}
                            onMouseLeave={(e) => e.currentTarget.style.background = "white"}
                          >
                            <span dangerouslySetInnerHTML={{ __html: p === "No results found" ? p : highlightMatch(p, protein) }} />
                          </li>
                        ))}
                      </ul>
                    )}
                  </label>
                  <div>
                    <button type="submit" disabled={loading} style={{ padding: "8px 16px" }}>
                      {loading ? "Predicting..." : "Predict"}
                    </button>
                  </div>
                </div>
              </form>

              {error && <div style={{ color: "#b00020", marginBottom: 16 }}>Error: {error}</div>}

              {result && (
                <div style={{ border: "1px solid #ddd", padding: 16, borderRadius: 8, marginBottom: 24 }}>
                  <h3>Result</h3>
                  <div><b>Drug:</b> {result.drug_name || drug}</div>
                  <div style={{ fontSize: 12, color: "#666", fontFamily: "monospace", wordBreak: "break-all" }}>
                    SMILES: {result.drug}
                  </div>
                  <div style={{ marginTop: 6 }}><b>Protein:</b> {result.protein}</div>
                  <div><b>Score:</b> {typeof result.score === "number" ? result.score.toFixed(4) : result.score}</div>
                  <div><b>Prediction:</b> {result.binder ? "Binder" : "Non-binder"}</div>
                  <div><b>Time:</b> {new Date(result.timestamp).toLocaleString()}</div>
                </div>
              )}
            </>
          )}

          {/* ===== VIRTUAL SCREENING MODE ===== */}
          {mode === "screen" && (
            <>
              <p style={{ margin: "0 0 12px", color: "#555", fontSize: 14 }}>
                Select multiple drugs and a single protein target to simulate a virtual screening assay.
                Results are ranked by predicted binding affinity.
              </p>

              <form onSubmit={onScreen}>
                {/* Compound library filters */}
                <div style={{ marginBottom: 20, border: "1px solid #e0e0e0", borderRadius: 8, padding: 16, background: "#fafafa" }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
                    <b style={{ fontSize: 14 }}>Compound Library Filters</b>
                    {hasActiveFilters && (
                      <button type="button" onClick={resetFilters}
                        style={{ padding: "3px 10px", fontSize: 12, cursor: "pointer", background: "none", border: "1px solid #ccc", borderRadius: 4, color: "#555" }}>
                        Reset Filters
                      </button>
                    )}
                  </div>

                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px 20px", marginBottom: 12 }}>
                    {/* Lipinski */}
                    <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13 }}>
                      <input type="checkbox" checked={filterLipinski} onChange={(e) => setFilterLipinski(e.target.checked)} />
                      <span>Lipinski's Rule of Five</span>
                      <span style={{ color: "#888", fontSize: 11 }}>(MW{"\u2264"}500, LogP{"\u2264"}5, HBA{"\u2264"}10, HBD{"\u2264"}5)</span>
                    </label>

                    {/* placeholder for grid alignment */}
                    <div />

                    {/* MW preset */}
                    <div style={{ fontSize: 13 }}>
                      <div style={{ marginBottom: 3 }}>Molecular Weight</div>
                      <select value={filterMwPreset} onChange={(e) => setFilterMwPreset(e.target.value)}
                        style={{ width: "100%", padding: 5 }}>
                        {MW_PRESETS.map((p) => (
                          <option key={p.value} value={p.value}>{p.label}</option>
                        ))}
                      </select>
                    </div>

                    {/* Ring count preset */}
                    <div style={{ fontSize: 13 }}>
                      <div style={{ marginBottom: 3 }}>Ring Count</div>
                      <select value={filterRingsPreset} onChange={(e) => setFilterRingsPreset(e.target.value)}
                        style={{ width: "100%", padding: 5 }}>
                        {RINGS_PRESETS.map((p) => (
                          <option key={p.value} value={p.value}>{p.label}</option>
                        ))}
                      </select>
                    </div>
                  </div>

                  {/* Known binders of target */}
                  <div style={{ marginBottom: 14, fontSize: 13 }}>
                    <div style={{ marginBottom: 3 }}>Known binders of target</div>
                    <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                      <div style={{ position: "relative", flex: 1 }}>
                        <input type="text" value={filterBinderTarget}
                          onChange={(e) => { setFilterBinderTarget(e.target.value); updateFilterBinderSuggestions(e.target.value) }}
                          placeholder="e.g., Epidermal growth factor receptor"
                          style={{ width: "100%", padding: 5 }} />
                        {filterBinderSuggestions.length > 0 && filterBinderTarget && (
                          <ul style={{
                            position: "absolute", left: 0, top: "100%", zIndex: 20, width: "100%",
                            border: "1px solid #888", padding: 0, background: "#fff", marginTop: 2,
                            listStyle: "none", borderRadius: 6, boxShadow: "0px 2px 6px rgba(0,0,0,0.15)",
                            maxHeight: 160, overflowY: "auto",
                          }}>
                            {filterBinderSuggestions.map((p, idx) => (
                              <li key={idx}
                                style={{ cursor: "pointer", padding: "5px 8px", borderBottom: "1px solid #eee", fontSize: 13 }}
                                onClick={() => { if (p !== "No results found") { setFilterBinderTarget(p); setFilterBinderSuggestions([]) } }}
                                onMouseEnter={(e) => e.currentTarget.style.background = "#f0f4ff"}
                                onMouseLeave={(e) => e.currentTarget.style.background = "white"}
                              >
                                <span dangerouslySetInnerHTML={{ __html: p === "No results found" ? p : highlightMatch(p, filterBinderTarget) }} />
                              </li>
                            ))}
                          </ul>
                        )}
                      </div>
                      {filterBinderTarget && (
                        <button type="button" onClick={() => setFilterBinderTarget("")}
                          style={{ background: "none", border: "none", cursor: "pointer", color: "#888", fontSize: 16, padding: "0 4px" }}>&times;</button>
                      )}
                    </div>
                  </div>

                  {/* Live count + add button */}
                  <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                    <button type="button" onClick={loadFilteredDrugs} disabled={filterLoading || !hasActiveFilters}
                      style={{
                        padding: "6px 16px", fontSize: 13, cursor: hasActiveFilters ? "pointer" : "default",
                        background: hasActiveFilters ? "#1a73e8" : "#ccc", color: "#fff", border: "none", borderRadius: 4,
                      }}>
                      {filterLoading ? "Loading..." : "Add Matching Compounds"}
                    </button>
                    {hasActiveFilters && (
                      <span style={{ fontSize: 13, color: "#555" }}>
                        {filterCountLoading
                          ? "counting..."
                          : filterCount != null
                            ? <><b>{filterCount.toLocaleString()}</b> compounds match</>
                            : null}
                      </span>
                    )}
                  </div>
                </div>

                {/* Drug multi-select */}
                <div style={{ marginBottom: 16 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                    <b>Selected Compounds</b>
                    <button type="button" onClick={loadRandomSample}
                      style={{ padding: "4px 10px", fontSize: 12, cursor: "pointer" }}>
                      + Random Sample (20)
                    </button>
                    {screenDrugs.length > 0 && (
                      <button type="button" onClick={() => setScreenDrugs([])}
                        style={{ padding: "4px 10px", fontSize: 12, cursor: "pointer" }}>
                        Clear All
                      </button>
                    )}
                  </div>

                  <input
                    type="text"
                    value={screenDrugQuery}
                    onChange={(e) => {
                      setScreenDrugQuery(e.target.value)
                      updateScreenDrugSuggestions(e.target.value)
                    }}
                    placeholder="Search drugs to add..."
                    style={{ width: "100%", padding: 8, marginBottom: 4 }}
                  />
                  {screenDrugSuggestions.length > 0 && screenDrugQuery && (
                    <ul style={{
                      border: "1px solid #888", padding: 0, background: "#fff", marginTop: 0,
                      listStyle: "none", borderRadius: 6, boxShadow: "0px 2px 6px rgba(0,0,0,0.15)",
                      maxHeight: 180, overflowY: "auto", position: "relative", zIndex: 10,
                    }}>
                      {screenDrugSuggestions.map((d, idx) => {
                        const display = d?.name || d?.smiles || "Unknown"
                        const isNoResult = display === "No results found"
                        const alreadyAdded = screenDrugs.some((existing) => existing.smiles === d.smiles)
                        return (
                          <li key={idx}
                            style={{
                              cursor: isNoResult ? "default" : "pointer",
                              padding: "6px 10px",
                              borderBottom: "1px solid #eee",
                              opacity: alreadyAdded ? 0.5 : 1,
                            }}
                            onClick={() => {
                              if (isNoResult || alreadyAdded) return
                              addScreenDrug({ name: d.name || d.smiles, smiles: d.smiles })
                            }}
                            onMouseEnter={(e) => (e.currentTarget.style.background = "#f0f4ff")}
                            onMouseLeave={(e) => (e.currentTarget.style.background = "white")}
                            title={d?.smiles ? `SMILES: ${d.smiles}` : ""}
                          >
                            <span dangerouslySetInnerHTML={{
                              __html: isNoResult ? display : highlightMatch(display, screenDrugQuery)
                            }} />
                            {alreadyAdded && <span style={{ marginLeft: 8, color: "#888", fontSize: 12 }}>(added)</span>}
                          </li>
                        )
                      })}
                    </ul>
                  )}

                  {/* Drug chips */}
                  {screenDrugs.length > 0 && (
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 8 }}>
                      {screenDrugs.map((d) => (
                        <span key={d.smiles} style={chipStyle} title={`SMILES: ${d.smiles}`}>
                          {d.name || d.smiles.substring(0, 20) + "..."}
                          <button type="button" onClick={() => removeScreenDrug(d.smiles)}
                            style={{ background: "none", border: "none", cursor: "pointer", fontSize: 14, padding: 0, lineHeight: 1 }}>
                            &times;
                          </button>
                        </span>
                      ))}
                    </div>
                  )}
                  <small style={{ color: "#888" }}>{screenDrugs.length} drug{screenDrugs.length !== 1 ? "s" : ""} selected (max 100)</small>
                </div>

                {/* Protein target */}
                <div style={{ marginBottom: 16 }}>
                  <b>Protein Target</b>
                  <input
                    type="text"
                    value={screenProtein}
                    onChange={(e) => {
                      setScreenProtein(e.target.value)
                      updateScreenProteinSuggestions(e.target.value)
                    }}
                    placeholder="e.g., Cytochrome P450 3A4"
                    style={{ width: "100%", padding: 8, marginTop: 4 }}
                  />
                  {screenProteinSuggestions.length > 0 && screenProtein && (
                    <ul style={{
                      border: "1px solid #888", padding: 0, background: "#fff", marginTop: 4,
                      listStyle: "none", borderRadius: 6, boxShadow: "0px 2px 6px rgba(0,0,0,0.15)",
                      maxHeight: 180, overflowY: "auto"
                    }}>
                      {screenProteinSuggestions.map((p, idx) => (
                        <li key={idx}
                          style={{ cursor: "pointer", padding: "6px 10px", borderBottom: "1px solid #eee" }}
                          onClick={() => { if (p !== "No results found") { setScreenProtein(p); setScreenProteinSuggestions([]) } }}
                          onMouseEnter={(e) => e.currentTarget.style.background = "#f0f4ff"}
                          onMouseLeave={(e) => e.currentTarget.style.background = "white"}
                        >
                          <span dangerouslySetInnerHTML={{ __html: p === "No results found" ? p : highlightMatch(p, screenProtein) }} />
                        </li>
                      ))}
                    </ul>
                  )}
                </div>

                <button type="submit" disabled={screenLoading} style={{ padding: "8px 20px" }}>
                  {screenLoading ? "Screening..." : `Screen ${screenDrugs.length} Drug${screenDrugs.length !== 1 ? "s" : ""}`}
                </button>
              </form>

              {screenError && <div style={{ color: "#b00020", marginTop: 12 }}>Error: {screenError}</div>}

              {/* Screening results table */}
              {screenResults && (
                <div style={{ marginTop: 20 }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4 }}>
                    <h3 style={{ margin: 0 }}>Screening Results</h3>
                    <button type="button" onClick={downloadCsv}
                      style={{ padding: "5px 14px", fontSize: 12, cursor: "pointer", border: "1px solid #ccc", borderRadius: 4, background: "#fff" }}>
                      Download CSV
                    </button>
                  </div>
                  <p style={{ color: "#555", fontSize: 13, margin: "0 0 10px" }}>
                    Target: <b>{screenResults.protein}</b> &mdash; {screenResults.hits.length} compounds screened &mdash; {new Date(screenResults.timestamp).toLocaleString()}
                  </p>
                  <div style={{ overflowX: "auto", position: "relative" }}>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
                      <thead>
                        <tr style={{ textAlign: "left" }}>
                          <th style={thStyle("rank")} onClick={() => handleSort("rank")}>Rank{sortIndicator("rank")}</th>
                          <th style={thStyle("drug_name")} onClick={() => handleSort("drug_name")}>Compound{sortIndicator("drug_name")}</th>
                          <th style={{ ...thStyle("drug"), cursor: "default" }}>SMILES</th>
                          <th style={thStyle("mw")} onClick={() => handleSort("mw")}>MW{sortIndicator("mw")}</th>
                          <th style={thStyle("logp")} onClick={() => handleSort("logp")}>LogP{sortIndicator("logp")}</th>
                          <th style={thStyle("rings")} onClick={() => handleSort("rings")}>Rings{sortIndicator("rings")}</th>
                          <th style={thStyle("score")} onClick={() => handleSort("score")}>Score{sortIndicator("score")}</th>
                          <th style={thStyle("binder")} onClick={() => handleSort("binder")}>Prediction{sortIndicator("binder")}</th>
                        </tr>
                      </thead>
                      <tbody>
                        {getSortedHits().map((hit) => (
                          <tr key={hit.drug} style={{
                            background: hit.binder ? "#e8f5e9" : "transparent",
                            borderBottom: "1px solid #eee",
                          }}>
                            <td style={{ padding: "6px 10px", fontWeight: 600 }}>{hit.rank}</td>
                            <td style={{ padding: "6px 10px", position: "relative" }}
                              onMouseEnter={(e) => {
                                const rect = e.currentTarget.getBoundingClientRect()
                                setHoverSmiles(hit.drug)
                                setHoverPos({ x: rect.right + 8, y: rect.top })
                              }}
                              onMouseLeave={() => setHoverSmiles(null)}
                            >
                              {hit.drug_name || <span style={{ color: "#999", fontStyle: "italic" }}>unnamed</span>}
                            </td>
                            <td style={{ padding: "6px 10px", fontFamily: "monospace", fontSize: 11, maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                              title={hit.drug}>
                              {hit.drug}
                            </td>
                            <td style={{ padding: "6px 10px", fontFamily: "monospace", fontSize: 12 }}>
                              {hit.mw != null ? hit.mw.toFixed(1) : "\u2013"}
                            </td>
                            <td style={{ padding: "6px 10px", fontFamily: "monospace", fontSize: 12 }}>
                              {hit.logp != null ? hit.logp.toFixed(2) : "\u2013"}
                            </td>
                            <td style={{ padding: "6px 10px", fontFamily: "monospace", fontSize: 12, textAlign: "center" }}>
                              {hit.rings != null ? hit.rings : "\u2013"}
                            </td>
                            <td style={{ padding: "6px 10px", fontFamily: "monospace" }}>
                              {hit.score.toFixed(4)}
                            </td>
                            <td style={{ padding: "6px 10px" }}>
                              <span style={{
                                display: "inline-block",
                                padding: "2px 8px",
                                borderRadius: 12,
                                fontSize: 12,
                                fontWeight: 600,
                                background: hit.binder ? "#4caf50" : "#e0e0e0",
                                color: hit.binder ? "#fff" : "#555",
                              }}>
                                {hit.binder ? "Binder" : "Non-binder"}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>

                    {/* Structure hover tooltip */}
                    {hoverSmiles && (
                      <div style={{
                        position: "fixed",
                        left: hoverPos.x,
                        top: hoverPos.y,
                        zIndex: 1000,
                        background: "#fff",
                        border: "1px solid #ccc",
                        borderRadius: 8,
                        boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
                        padding: 6,
                        pointerEvents: "none",
                      }}>
                        <img
                          src={`${API_BASE}/structure/svg?smiles=${encodeURIComponent(hoverSmiles)}&w=220&h=180`}
                          alt="2D structure"
                          width={220}
                          height={180}
                          style={{ display: "block" }}
                        />
                      </div>
                    )}
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* History (shown in both modes) */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <h2 style={{ margin: 0 }}>Previous Searches</h2>
          <button onClick={onClearHistory} style={{ padding: "6px 12px"}}>Clear</button>
        </div>
        <ul className={styles.list}>
          {history.length === 0 && <li>No searches yet.</li>}
          {history.map((h) => (
            <li key={h.id || h.timestamp} className={styles.listItem} style={{ display: "flex", flexDirection: "column" }}>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                <span title={`SMILES: ${h.drug}`}><b>Drug:</b> {h.drug_name || h.drug}</span>
                <span><b>Protein:</b> {h.protein}</span>
                <span><b>Score:</b> {typeof h.score === "number" ? h.score.toFixed(4) : h.score}</span>
                <span><b>Prediction:</b> {h.binder ? "Binder" : "Non-binder"}</span>
              </div>
              <small style={{ color: "#666" }}>{new Date(h.timestamp).toLocaleString()}</small>
            </li>
          ))}
        </ul>
      </div>
    </Layout>
  )
}

export const Head = () => <Seo title="DTI Checker" />

export default IndexPage
