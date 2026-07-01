import os, re, json, httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

load_dotenv()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE = "https://api.groq.com/openai/v1/chat/completions"

# Changed from llama-3.3-70b-versatile to GPT-OSS-120B
GROQ_MODEL = "openai/gpt-oss-120b"

@app.get("/health")
def health():
    return {"status": "ok", "tool": "TargetScope", "ai": "openai/gpt-oss-120b"}

@app.get("/opentargets")
async def opentargets(disease: str):
    """Query OpenTargets GraphQL for disease-target associations"""
    query = """
    query diseaseTargets($efoId: String!) {
      disease(efoId: $efoId) {
        name
        associatedTargets(page: {index: 0, size: 15}) {
          rows {
            score
            target {
              id
              approvedSymbol
              approvedName
              biotype
              functionDescriptions
            }
          }
        }
      }
    }
    """
    result = {"targets": [], "disease_name": disease, "efo_id": ""}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            # First search for disease EFO ID
            search_resp = await client.get(
                "https://api.platform.opentargets.org/api/v4/graphql",
                params={"query": f"{{ search(queryString: \"{disease}\", entityNames: [\"disease\"]) {{ hits {{ id name entity }} }} }}"}
            )
            # Use REST search instead
            search_resp2 = await client.get(
                f"https://api.platform.opentargets.org/api/v4/graphql",
                params={"query": "{ search(queryString: \"" + disease + "\", entityNames: [\"disease\"]) { hits { id name } } }"}
            )
            if search_resp2.status_code == 200:
                hits = search_resp2.json().get("data", {}).get("search", {}).get("hits", [])
                if hits:
                    efo_id = hits[0]["id"]
                    result["efo_id"] = efo_id
                    result["disease_name"] = hits[0]["name"]
                    # Now get targets
                    gql_resp = await client.post(
                        "https://api.platform.opentargets.org/api/v4/graphql",
                        json={"query": query, "variables": {"efoId": efo_id}},
                        headers={"Content-Type": "application/json"}
                    )
                    if gql_resp.status_code == 200:
                        data = gql_resp.json().get("data", {}).get("disease", {})
                        rows = data.get("associatedTargets", {}).get("rows", [])
                        for r in rows:
                            t = r.get("target", {})
                            result["targets"].append({
                                "symbol": t.get("approvedSymbol", ""),
                                "name": t.get("approvedName", ""),
                                "biotype": t.get("biotype", ""),
                                "score": round(r.get("score", 0), 3),
                                "functions": (t.get("functionDescriptions") or [""])[:1]
                            })
    except Exception as e:
        result["error"] = str(e)
    return result

@app.get("/pubmed-targets")
async def pubmed_targets(disease: str):
    """Get recent target-focused papers from PubMed"""
    results = []
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            search = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={"db": "pubmed", "term": f"{disease} drug target novel 2022:2025[dp]", "retmax": 8, "retmode": "json", "sort": "relevance"}
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
                for i, t in enumerate(titles[:8]):
                    results.append({
                        "title": re.sub(r'<[^>]+>', '', t).strip(),
                        "abstract": re.sub(r'<[^>]+>', '', abstracts[i] if i < len(abstracts) else "").strip()[:400],
                        "year": years[i] if i < len(years) else ""
                    })
    except Exception as e:
        results = [{"error": str(e)}]
    return results

from pydantic import BaseModel
class AnalyzeRequest(BaseModel):
    disease: str
    targets: list
    papers: list
    context: str = ""

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not configured"}

    targets_str = json.dumps(req.targets[:12], indent=2)
    papers_str = json.dumps([{"title": p.get("title",""), "abstract": p.get("abstract","")} for p in req.papers[:6]], indent=2)

    prompt = f"""You are an expert computational biologist and drug discovery scientist. Analyze the following disease target data and recent literature to provide actionable drug target discovery insights.

Disease/Condition: {req.disease}
Research Context: {req.context or "General drug discovery"}

OpenTargets Association Data (top targets by evidence score):
{targets_str}

Recent Literature Highlights:
{papers_str}

Respond ONLY with a raw JSON object. No markdown. No code fences.

{{
  "top_targets": [
    {{
      "symbol": "gene symbol",
      "name": "full name",
      "priority": "High/Medium/Low",
      "rationale": "2-sentence mechanistic rationale for this target",
      "pathway": "key biological pathway",
      "novelty": "Established/Emerging/Novel",
      "druggability": "High/Moderate/Low",
      "therapeutic_modality": "small molecule/antibody/gene therapy/etc"
    }}
  ],
  "key_pathways": ["pathway1", "pathway2", "pathway3", "pathway4"],
  "disease_biology_summary": "3-sentence summary of the key biological mechanisms underlying this disease",
  "target_landscape": "2-sentence overview of the current target landscape",
  "high_value_combinations": ["target pair or combination rationale 1", "target pair 2"],
  "gaps_and_opportunities": "2-sentence description of underexplored target space",
  "confidence": integer 0-100,
  "data_richness": "Rich/Moderate/Sparse"
}}"""

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            GROQ_BASE,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "messages": [
                {"role": "system", "content": "You are a JSON API for drug target discovery. Output only raw JSON. No markdown. No explanation."},
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
