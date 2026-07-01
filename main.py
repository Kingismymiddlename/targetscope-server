import os, re, json, httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE = "https://api.groq.com/openai/v1/chat/completions"

# Changed from llama-3.3-70b-versatile to GPT-OSS-120B
GROQ_MODEL = "openai/gpt-oss-120b"

@app.get("/health")
def health():
    return {"status": "ok", "tool": "BioSignal", "ai": "openai/gpt-oss-120b"}

@app.get("/pubmed-biomarkers")
async def pubmed_biomarkers(disease: str, biomarker_type: str = "diagnostic"):
    results = []
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            query = f"{disease} biomarker {biomarker_type} 2020:2025[dp]"
            search = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={"db": "pubmed", "term": query, "retmax": 10, "retmode": "json", "sort": "relevance"}
            )
            ids = search.json().get("esearchresult", {}).get("idlist", [])
            if ids:
                fetch = await client.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                    params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml", "rettype": "abstract"}
                )
                xml = fetch.text
                titles = re.findall(r'<ArticleTitle>(.*?)</ArticleTitle>', xml, re.DOTALL)
                abstracts = re.findall(r'<AbstractText[^>]*>(.*?)</AbstractText>', xml, re.DOTALL)
                years = re.findall(r'<PubDate>.*?<Year>(\d{4})</Year>', xml, re.DOTALL)
                journals = re.findall(r'<Title>(.*?)</Title>', xml, re.DOTALL)
                for i, t in enumerate(titles[:10]):
                    results.append({
                        "title": re.sub(r'<[^>]+>', '', t).strip(),
                        "abstract": re.sub(r'<[^>]+>', '', abstracts[i] if i < len(abstracts) else "").strip()[:500],
                        "year": years[i] if i < len(years) else "",
                        "journal": journals[i] if i < len(journals) else ""
                    })
    except Exception as e:
        results = [{"error": str(e)}]
    return results

@app.get("/disgenet")
async def disgenet(disease: str):
    """Query DisGeNET via UMLS for gene-disease associations"""
    results = []
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                "https://www.disgenet.org/api/gda/disease/search",
                params={"disease": disease, "limit": 15, "source": "ALL"},
                headers={"accept": "application/json"}
            )
            if resp.status_code == 200:
                for item in resp.json()[:15]:
                    results.append({
                        "gene": item.get("gene_symbol", ""),
                        "gene_name": item.get("gene_name", ""),
                        "score": item.get("score", 0),
                        "pmids": item.get("pmid_count", 0),
                        "disease_name": item.get("disease_name", "")
                    })
    except Exception as e:
        results = [{"error": str(e)}]
    return results

class BiomarkerRequest(BaseModel):
    disease: str
    biomarker_type: str = "diagnostic"
    patient_context: str = ""
    papers: list = []
    genes: list = []

@app.post("/analyze")
async def analyze(req: BiomarkerRequest):
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not configured"}

    papers_str = json.dumps([{"title": p.get("title",""), "abstract": p.get("abstract","")} for p in req.papers[:8]], indent=2)
    genes_str = json.dumps(req.genes[:12], indent=2)

    prompt = f"""You are an expert molecular biologist and clinical diagnostics scientist specializing in biomarker discovery. Analyze the following data to identify high-value biomarkers.

Disease/Condition: {req.disease}
Biomarker Type Requested: {req.biomarker_type}
Patient/Clinical Context: {req.patient_context or "General population"}

Gene-Disease Association Data:
{genes_str}

Recent Biomarker Literature:
{papers_str}

Respond ONLY with a raw JSON object. No markdown. No code fences.

{{
  "biomarkers": [
    {{
      "name": "biomarker name (gene/protein/metabolite)",
      "type": "Protein/Gene/Metabolite/miRNA/Cytokine/Antibody",
      "role": "Diagnostic/Prognostic/Predictive/Theranostic",
      "clinical_value": "High/Medium/Low",
      "specimen": "Blood/Urine/CSF/Tissue/Saliva",
      "rationale": "2-sentence explanation of why this is a valuable biomarker",
      "current_status": "Validated/Emerging/Investigational",
      "sensitivity_specificity": "brief note on known diagnostic performance if available",
      "assay_type": "ELISA/PCR/NGS/Mass spec/Flow cytometry/etc"
    }}
  ],
  "disease_summary": "2-sentence overview of the biomarker landscape for this condition",
  "best_panel": {{
    "name": "Suggested multi-biomarker panel name",
    "components": ["biomarker1", "biomarker2", "biomarker3"],
    "rationale": "Why this combination provides optimal diagnostic/prognostic value"
  }},
  "clinical_challenges": "2-sentence description of key challenges in biomarker use for this disease",
  "emerging_technologies": ["technology1 relevant to biomarker detection", "technology2"],
  "regulatory_landscape": "1-sentence note on FDA/EMA approval status of biomarkers in this space",
  "confidence": integer 0-100,
  "evidence_grade": "A/B/C/D (A=multiple RCTs, B=cohort studies, C=case series, D=expert opinion)"
}}"""

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            GROQ_BASE,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "messages": [
                {"role": "system", "content": "You are a JSON API for biomarker discovery. Output only raw JSON. No markdown."},
                {"role": "user", "content": prompt}
            ], "temperature": 0.2, "max_tokens": 2000, "response_format": {"type": "json_object"}}
        )
        data = resp.json()
        if "error" in data:
            return {"error": data["error"].get("message", "Groq error")}
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        text = re.sub(r'^```json\s*', '', text); text = re.sub(r'^```\s*', '', text); text = re.sub(r'\s*```$', '', text).strip()
        try:
            return json.loads(text)
        except:
            m = re.search(r'\{[\s\S]*\}', text)
            if m:
                try: return json.loads(m.group())
                except: pass
        return {"error": f"Parse error: {text[:200]}"}

app.mount("/", StaticFiles(directory="static", html=True), name="static")
